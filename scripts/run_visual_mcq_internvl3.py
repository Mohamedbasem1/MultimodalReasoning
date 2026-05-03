import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from run_visual_mcq_qwen25 import (
    ANSWER_KEYS,
    DEFAULT_DATASET,
    load_prompt,
    normalize_image,
    parse_answer,
    pick_column,
)
from run_visual_mcq_voting import build_image_variants


DEFAULT_MODEL = "OpenGVLab/InternVL3-8B-hf"
DEFAULT_PROMPT = """You are solving a visual multiple-choice exam question.

Read the image carefully, including diagrams, charts, equations, labels, units, and all answer options.

Think internally if needed, but output only the final option letter: A, B, C, D, or E.
Do not output explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InternVL3-8B on Visual MCQ datasets.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_internvl3_8b.json")
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path for raw model outputs. Defaults to '<output>.raw.jsonl'.",
    )
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
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
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading if available.",
    )
    parser.add_argument(
        "--image-variant",
        default="enhanced",
        choices=["original", "enhanced"],
        help="Image variant sent to InternVL3.",
    )
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    return parser.parse_args()


def dtype_from_arg(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def build_inputs(processor: AutoProcessor, image: Image.Image, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    move_kwargs: Dict[str, Any] = {"device": device}
    if device.type == "cuda":
        move_kwargs["dtype"] = torch.bfloat16
    return inputs.to(**move_kwargs)


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
        "torch_dtype": dtype_from_arg(args.torch_dtype),
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    return AutoModelForImageTextToText.from_pretrained(args.model, **kwargs).eval()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. InternVL3-8B inference will be very slow on CPU.")

    prompt = load_prompt(args.prompt_file) if Path(args.prompt_file).exists() else DEFAULT_PROMPT

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
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_model(args)
    device = model_device(model)

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="InternVL3"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            image = build_image_variants(
                image,
                requested_variants=[args.image_variant],
                enhance_longest_side=args.enhance_longest_side,
            )[0][1]
            inputs = build_inputs(processor, image, prompt, device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )

            raw_text = processor.decode(
                generated_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            answer_key = parse_answer(raw_text, args.fallback_answer)
            predictions.append({"question_id": question_id, "answer_key": answer_key})
            raw_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "answer_key": answer_key,
                        "raw_text": raw_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if has_gold:
                gold = str(row[args.answer_column]).strip().upper()
                if gold in ANSWER_KEYS:
                    scored += 1
                    correct += int(answer_key == gold)

    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw outputs: {raw_output_path}")
    if scored:
        print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")


if __name__ == "__main__":
    main()

