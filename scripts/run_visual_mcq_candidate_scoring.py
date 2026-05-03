import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from run_visual_mcq_qwen25 import (
    ANSWER_KEYS,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    load_model,
    load_prompt,
    model_device,
    normalize_image,
    pick_column,
)
from run_visual_mcq_voting import (
    BUILTIN_PROMPTS,
    augment_prompt_with_ocr,
    build_image_variants,
    load_ocr_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score A/B/C/D/E candidates by log probability for ImageCLEF Visual MCQ."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT/LoRA adapter directory produced by train_visual_mcq_lora.py.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_qwen25vl7b_scored.json")
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path for raw candidate scores. Defaults to '<output>.raw.jsonl'.",
    )
    parser.add_argument(
        "--prompt-files",
        nargs="+",
        default=None,
        help="Optional prompt files. If omitted, uses the built-in direct/OCR/verify prompt variants.",
    )
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--filter-type",
        nargs="+",
        default=None,
        help="Optional values for the dataset 'type' column, e.g. image image_text.",
    )
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--fallback-answer", choices=sorted(ANSWER_KEYS), default="A")
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=[None, "flash_attention_2", "sdpa", "eager"],
        help="Use flash_attention_2 on Linux if installed; otherwise leave unset.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading. Best on Linux; may not work on Windows.",
    )
    parser.add_argument(
        "--image-variants",
        nargs="+",
        default=["original"],
        choices=["original", "enhanced"],
        help="Score one or more image variants. Use 'original enhanced' to ensemble over both.",
    )
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    parser.add_argument(
        "--ocr-json",
        default=None,
        help="Optional JSON/JSONL with precomputed question_id and ocr_text records.",
    )
    parser.add_argument("--ocr-max-chars", type=int, default=1800)
    parser.add_argument(
        "--aggregation",
        default="sum",
        choices=["sum", "vote"],
        help="Aggregate per-prompt/per-variant candidate scores by summed logprob or majority vote.",
    )
    parser.add_argument(
        "--score-full-suffix",
        action="store_true",
        help="Score all tokens after the assistant prompt, including end-of-turn tokens. Default scores only the answer letter token(s).",
    )
    return parser.parse_args()


def load_prompts(prompt_files: Optional[List[str]], num_prompts: Optional[int]) -> List[Tuple[str, str]]:
    if prompt_files:
        prompts = []
        for prompt_file in prompt_files:
            prompt_path = Path(prompt_file)
            prompts.append((prompt_path.stem, load_prompt(prompt_file)))
    else:
        prompts = BUILTIN_PROMPTS.copy()

    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    if not prompts:
        raise ValueError("At least one prompt is required.")
    return prompts


def build_texts(processor: AutoProcessor, image: Image.Image, prompt: str, answer: str) -> Tuple[str, str]:
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


def candidate_token_len(processor: AutoProcessor, answer: str) -> int:
    token_ids = processor.tokenizer(answer, add_special_tokens=False).input_ids
    return max(1, len(token_ids))


def score_candidate(
    model: torch.nn.Module,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    answer: str,
    score_full_suffix: bool,
) -> float:
    prompt_text, full_text = build_texts(processor, image, prompt, answer)
    full_inputs = processor(
        text=[full_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    full_inputs = full_inputs.to(model_device(model))
    prompt_inputs = prompt_inputs.to(model_device(model))

    input_ids = full_inputs.input_ids[0]
    prompt_len = int(prompt_inputs.attention_mask[0].sum().item())
    full_len = int(full_inputs.attention_mask[0].sum().item())
    if score_full_suffix:
        target_end = full_len
    else:
        target_end = min(full_len, prompt_len + candidate_token_len(processor, answer))

    if prompt_len >= target_end:
        return float("-inf")

    with torch.inference_mode():
        logits = model(**full_inputs).logits[0]

    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    token_positions = torch.arange(prompt_len, target_end, device=logits.device)
    score_positions = token_positions - 1
    target_ids = input_ids[token_positions]
    token_scores = log_probs[score_positions, target_ids]
    return float(token_scores.sum().item())


def choose_answer(
    score_records: List[Dict[str, Any]],
    fallback_answer: str,
    aggregation: str,
) -> Tuple[str, Dict[str, float]]:
    totals = {answer: 0.0 for answer in sorted(ANSWER_KEYS)}
    if aggregation == "sum":
        for record in score_records:
            for answer, score in record["scores"].items():
                totals[answer] += float(score)
        return max(totals.items(), key=lambda item: item[1])[0], totals

    votes = []
    for record in score_records:
        scores = record["scores"]
        votes.append(max(scores.items(), key=lambda item: item[1])[0])
    if not votes:
        return fallback_answer, totals
    vote_counts = {answer: votes.count(answer) for answer in sorted(ANSWER_KEYS)}
    top_count = max(vote_counts.values())
    tied = {answer for answer, count in vote_counts.items() if count == top_count}
    for record in score_records:
        winner = max(record["scores"].items(), key=lambda item: item[1])[0]
        if winner in tied:
            return winner, {answer: float(count) for answer, count in vote_counts.items()}
    return fallback_answer, {answer: float(count) for answer, count in vote_counts.items()}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Candidate scoring will be very slow on CPU.")

    prompts = load_prompts(args.prompt_files, args.num_prompts)
    print("Prompts:")
    for name, _prompt in prompts:
        print(f"- {name}")
    print(f"Image variants: {', '.join(args.image_variants)}")
    print(f"Aggregation: {args.aggregation}")

    ocr_lookup = load_ocr_json(args.ocr_json, args.ocr_max_chars)
    if ocr_lookup:
        print(f"Loaded precomputed OCR records: {len(ocr_lookup)}")

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
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = load_model(args)
    model.eval()

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="Candidate scoring"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            image_variants = build_image_variants(
                image,
                requested_variants=args.image_variants,
                enhance_longest_side=args.enhance_longest_side,
            )
            ocr_text = ocr_lookup.get(question_id, "")
            score_records: List[Dict[str, Any]] = []

            for variant_name, variant_image in image_variants:
                for prompt_name, prompt in prompts:
                    final_prompt = augment_prompt_with_ocr(prompt, ocr_text)
                    candidate_scores = {}
                    for answer in sorted(ANSWER_KEYS):
                        candidate_scores[answer] = score_candidate(
                            model=model,
                            processor=processor,
                            image=variant_image,
                            prompt=final_prompt,
                            answer=answer,
                            score_full_suffix=args.score_full_suffix,
                        )
                    score_records.append(
                        {
                            "variant": variant_name,
                            "prompt": prompt_name,
                            "scores": candidate_scores,
                            "winner": max(candidate_scores.items(), key=lambda item: item[1])[0],
                        }
                    )

            answer_key, aggregate_scores = choose_answer(
                score_records,
                fallback_answer=args.fallback_answer,
                aggregation=args.aggregation,
            )
            predictions.append({"question_id": question_id, "answer_key": answer_key})

            raw_record: Dict[str, Any] = {
                "question_id": question_id,
                "answer_key": answer_key,
                "aggregate_scores": aggregate_scores,
                "ocr_text": ocr_text,
                "score_records": score_records,
            }
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
    print(f"Wrote raw scores: {raw_output_path}")
    if scored:
        print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")


if __name__ == "__main__":
    main()

