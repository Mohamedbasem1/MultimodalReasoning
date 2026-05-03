import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor

from run_visual_mcq_qwen25 import (
    ANSWER_KEYS,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    load_model,
    normalize_image,
    pick_column,
)
from run_visual_mcq_voting import BUILTIN_PROMPTS, build_image_variants, generate_one, vote_answers


VERIFIER_TEMPLATE = """You are verifying a visual multiple-choice exam question.

Independent passes proposed these option letters:
{candidate_lines}

Re-read the image carefully, including all text, answer options, diagrams, charts, labels, equations, and units.
If the proposals disagree, do not vote mechanically. Choose the option best supported by the image.

Output only one uppercase option letter: A, B, C, D, or E.
Do not explain your reasoning."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-VL MCQ self-consistency with verifier on disagreement."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_qwen25vl7b_consistency.json")
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--fallback-answer", choices=sorted(ANSWER_KEYS), default="A")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    parser.add_argument(
        "--agreement-policy",
        default="unanimous_then_verifier",
        choices=["unanimous_then_verifier", "majority_then_verifier", "always_verifier"],
    )
    parser.add_argument("--primary-on-verifier-fail", action="store_true")
    return parser.parse_args()


def verifier_prompt(votes: List[Dict[str, str]]) -> str:
    candidate_lines = "\n".join(f"- {vote['prompt']}: {vote['answer_key']}" for vote in votes)
    return VERIFIER_TEMPLATE.format(candidate_lines=candidate_lines)


def needs_verifier(votes: List[Dict[str, str]], policy: str) -> bool:
    answers = [vote["answer_key"] for vote in votes]
    if policy == "always_verifier":
        return True
    if policy == "unanimous_then_verifier":
        return len(set(answers)) > 1
    counts = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.values()) < 2


def choose_answer(
    votes: List[Dict[str, str]],
    verifier_answer: Optional[str],
    fallback_answer: str,
    policy: str,
    primary_on_verifier_fail: bool,
) -> Tuple[str, str, Dict[str, int]]:
    vote_answer, vote_counts = vote_answers(votes, fallback_answer)
    primary_answer = votes[0]["answer_key"] if votes else fallback_answer
    if not needs_verifier(votes, policy):
        if policy == "unanimous_then_verifier":
            return primary_answer, "unanimous", vote_counts
        return vote_answer, "majority", vote_counts
    if verifier_answer:
        return verifier_answer, "verifier", vote_counts
    return (primary_answer if primary_on_verifier_fail else vote_answer), "fallback", vote_counts


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.filter_type:
        allowed_types = set(args.filter_type)
        dataset = dataset.filter(lambda row: row.get("type") in allowed_types)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    image_column = pick_column(dataset.column_names, args.image_column, ["image", "image_id"])
    has_gold = args.answer_column in dataset.column_names
    print(f"Rows: {len(dataset)}")
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")
    if has_gold:
        print(f"Gold column: {args.answer_column}")

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = load_model(args)
    model.eval()

    prompts = BUILTIN_PROMPTS[:3]
    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0
    decision_counts: Dict[str, int] = {}

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="Consistency"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            variant_image = build_image_variants(
                image,
                requested_variants=[args.image_variant],
                enhance_longest_side=args.enhance_longest_side,
            )[0][1]

            votes = []
            for prompt_name, prompt in prompts:
                answer_key, raw_text = generate_one(
                    model=model,
                    processor=processor,
                    image=variant_image,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    fallback_answer=args.fallback_answer,
                )
                votes.append({"prompt": prompt_name, "answer_key": answer_key, "raw_text": raw_text})

            run_verifier = needs_verifier(votes, args.agreement_policy)
            verifier_answer = None
            verifier_raw = ""
            if run_verifier:
                verifier_answer, verifier_raw = generate_one(
                    model=model,
                    processor=processor,
                    image=variant_image,
                    prompt=verifier_prompt(votes),
                    max_new_tokens=args.verifier_max_new_tokens,
                    fallback_answer=args.fallback_answer,
                )

            answer_key, decision_mode, vote_counts = choose_answer(
                votes=votes,
                verifier_answer=verifier_answer,
                fallback_answer=args.fallback_answer,
                policy=args.agreement_policy,
                primary_on_verifier_fail=args.primary_on_verifier_fail,
            )
            decision_counts[decision_mode] = decision_counts.get(decision_mode, 0) + 1
            predictions.append({"question_id": question_id, "answer_key": answer_key})

            raw_record: Dict[str, Any] = {
                "question_id": question_id,
                "answer_key": answer_key,
                "decision_mode": decision_mode,
                "vote_counts": vote_counts,
                "votes": votes,
            }
            if run_verifier:
                raw_record["verifier"] = {"answer_key": verifier_answer, "raw_text": verifier_raw}
            if has_gold:
                raw_record["gold"] = str(row[args.answer_column]).strip().upper()
            raw_file.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

            if has_gold:
                gold = str(row[args.answer_column]).strip().upper()
                if gold in ANSWER_KEYS:
                    scored += 1
                    correct += int(answer_key == gold)

    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw outputs: {raw_output_path}")
    print(f"Decision modes: {decision_counts}")
    if scored:
        print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")


if __name__ == "__main__":
    main()

