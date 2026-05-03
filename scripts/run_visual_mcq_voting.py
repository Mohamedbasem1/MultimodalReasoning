import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor

from run_visual_mcq_qwen25 import (
    ANSWER_KEYS,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    build_inputs,
    load_model,
    load_prompt,
    model_device,
    normalize_image,
    parse_answer,
    pick_column,
)


BUILTIN_PROMPTS: List[Tuple[str, str]] = [
    (
        "direct",
        """You are solving a multiple-choice exam question from an image.

Read the full image carefully, including all question text, answer options, diagrams, charts, tables, labels, formulas, and units.

Choose exactly one correct option.

Return only one uppercase letter: A, B, C, D, or E.
Do not explain your reasoning.""",
    ),
    (
        "ocr",
        """You are solving a multiple-choice exam question from an image.

First, carefully read every visible word, number, option label, formula, table cell, chart axis, diagram label, and unit in the image.

Use the visual evidence to choose the best answer option.

Return only one uppercase letter: A, B, C, D, or E.
Do not explain your reasoning.""",
    ),
    (
        "verify",
        """You are solving a multiple-choice exam question from an image.

Read the question and all answer options from the image. Compare the options against the image evidence, eliminate wrong options, and choose the most correct remaining option.

Return only one uppercase letter: A, B, C, D, or E.
Do not explain your reasoning.""",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-prompt voting inference for ImageCLEF 2026 Visual MCQ."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT/LoRA adapter directory produced by train_visual_mcq_lora.py.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_qwen25vl7b_voting.json")
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path for raw vote outputs. Defaults to '<output>.raw.jsonl'.",
    )
    parser.add_argument(
        "--prompt-files",
        nargs="+",
        default=None,
        help="Optional prompt files. If omitted, uses three built-in prompt variants.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Use only the first N prompts. Useful for fast tests.",
    )
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
    parser.add_argument("--max-new-tokens", type=int, default=8)
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


def vote_answers(votes: List[Dict[str, str]], fallback: str) -> Tuple[str, Dict[str, int]]:
    counts = Counter(vote["answer_key"] for vote in votes)
    if not counts:
        return fallback, {}

    top_count = max(counts.values())
    tied_answers = {answer for answer, count in counts.items() if count == top_count}
    if len(tied_answers) == 1:
        return next(iter(tied_answers)), dict(sorted(counts.items()))

    for vote in votes:
        if vote["answer_key"] in tied_answers:
            return vote["answer_key"], dict(sorted(counts.items()))
    return fallback, dict(sorted(counts.items()))


def generate_one(
    model: torch.nn.Module,
    processor: AutoProcessor,
    image: Any,
    prompt: str,
    max_new_tokens: int,
    fallback_answer: str,
) -> Tuple[str, str]:
    inputs = build_inputs(processor, image, prompt)
    inputs = inputs.to(model_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_answer(raw_text, fallback_answer), raw_text


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Voting inference will be very slow on CPU.")

    prompts = load_prompts(args.prompt_files, args.num_prompts)
    print("Prompts:")
    for name, _prompt in prompts:
        print(f"- {name}")

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
        for row in tqdm(dataset, desc="Voting"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            votes = []

            for prompt_name, prompt in prompts:
                answer_key, raw_text = generate_one(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    fallback_answer=args.fallback_answer,
                )
                votes.append(
                    {
                        "prompt": prompt_name,
                        "answer_key": answer_key,
                        "raw_text": raw_text,
                    }
                )

            voted_answer, vote_counts = vote_answers(votes, args.fallback_answer)
            predictions.append({"question_id": question_id, "answer_key": voted_answer})

            raw_record: Dict[str, Any] = {
                "question_id": question_id,
                "answer_key": voted_answer,
                "vote_counts": vote_counts,
                "votes": votes,
            }
            if has_gold:
                raw_record["gold"] = str(row[args.answer_column]).strip().upper()
            raw_file.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

            if has_gold:
                gold = str(row[args.answer_column]).strip().upper()
                if gold in ANSWER_KEYS:
                    scored += 1
                    correct += int(voted_answer == gold)

    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw votes: {raw_output_path}")
    if scored:
        print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")


if __name__ == "__main__":
    main()

