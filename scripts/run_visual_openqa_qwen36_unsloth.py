import argparse
import json
import os
import re
import builtins
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


os.environ.setdefault("UNSLOTH_MOE_BACKEND", "native_torch")


def patch_unsloth_transformers_symbols() -> None:
    """Provide symbols Unsloth expects while patching new Transformers models."""
    try:
        import huggingface_hub
        if not hasattr(huggingface_hub, "is_offline_mode"):
            def is_offline_mode() -> bool:
                value = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE") or ""
                return value.upper() in {"1", "ON", "YES", "TRUE"}

            huggingface_hub.is_offline_mode = is_offline_mode
    except Exception:
        pass
    try:
        from transformers.utils import auto_docstring
    except Exception:
        def auto_docstring(obj=None, *args, **kwargs):
            if callable(obj):
                return obj

            def decorator(inner):
                return inner

            return decorator
    builtins.auto_docstring = auto_docstring
    try:
        from huggingface_hub.dataclasses import strict
    except Exception:
        def strict(obj=None, *args, **kwargs):
            if callable(obj):
                return obj

            def decorator(inner):
                return inner

            return decorator
    builtins.strict = strict
    try:
        from transformers.utils.type_validators import interval
    except Exception:
        def interval(*args, default=None, **kwargs):
            return default
    builtins.interval = interval
    try:
        from transformers import PreTrainedConfig
    except Exception:
        from transformers import PretrainedConfig as PreTrainedConfig
    builtins.PreTrainedConfig = PreTrainedConfig
    builtins.PretrainedConfig = PreTrainedConfig
    try:
        from transformers.modeling_rope_utils import RopeParameters
        builtins.RopeParameters = RopeParameters
    except Exception:
        pass


patch_unsloth_transformers_symbols()
import unsloth  # noqa: F401
import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm


DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"
DEFAULT_PROMPT = """Answer the visual open-ended exam question in the image.

Read all visible question text, diagrams, charts, tables, labels, formulas, and units.

Output only the final answer, in the same language as the question when possible.

Use concise exam-answer style. If the question has parts, answer as A), B), C) or 1), 2), 3).

Do not describe the image. Do not say what the user wants. Do not explain your reasoning.
Do not output chain-of-thought, markdown analysis, bullet planning, or <think> tags."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental Unsloth inference for Qwen3.6-35B-A3B Visual OpenQA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="Optional Unsloth/PEFT LoRA adapter directory.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_openqa_qwen36_35b_a3b_unsloth.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--prompt-file", default="prompts/visual_openqa_prompt.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="auto")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-answer-chars", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--answer-prefill", default="FINAL ANSWER:\n")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1000)
    return parser.parse_args()


def load_prompt(path: str) -> str:
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def pick_column(columns: Sequence[str], requested: str, candidates: Iterable[str], required: bool = True) -> str:
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"Column '{requested}' not found. Available: {list(columns)}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"Could not infer column. Available: {list(columns)}")
    return ""


def normalize_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        from io import BytesIO

        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def resize_longest_side(image: Image.Image, longest_side: int) -> Image.Image:
    if longest_side <= 0:
        return image
    width, height = image.size
    current_longest = max(width, height)
    if current_longest == 0 or current_longest == longest_side:
        return image
    scale = longest_side / current_longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def enhance_image(image: Image.Image, longest_side: int) -> Image.Image:
    enhanced = resize_longest_side(image, longest_side)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.35)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))


def select_image_variant(image: Image.Image, variant: str, longest_side: int) -> Image.Image:
    if variant == "original":
        return image
    if variant == "enhanced":
        return enhance_image(image, longest_side)
    raise ValueError(f"Unsupported image variant: {variant}")


def clean_answer(raw_text: str, max_chars: int) -> str:
    text = raw_text.strip()
    if re.search(r"final\s+answer\s*:", text, flags=re.IGNORECASE):
        text = re.split(r"final\s+answer\s*:", text, flags=re.IGNORECASE)[-1]
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    else:
        text = re.sub(r"<think>.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<answer>|</answer>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:final\s+answer|answer)\s*(?:is|:|-)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:the\s+)?user\s+wants\s+me\s+to\s+[^.:\n]*(?:\.|:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:image\s+analysis|problem\s+analysis|analysis)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n\"'")
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def normalize_for_match(text: Any) -> str:
    value = str(text).lower()
    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.DOTALL)
    value = re.sub(r"[^\w\s.%/-]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def build_inputs(processor: Any, image: Image.Image, prompt: str, answer_prefill: str) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    input_text += answer_prefill
    return processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastVisionModel
    except ImportError as exc:
        raise ImportError(
            "Missing Unsloth stack. Install with: pip install -r requirements-unsloth-qwen36.txt"
        ) from exc

    prompt = load_prompt(args.prompt_file)
    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.filter_type:
        allowed_types = set(args.filter_type)
        dataset = dataset.filter(lambda row: row.get("type") in allowed_types)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    image_column = pick_column(dataset.column_names, args.image_column, ["image", "image_id"])
    answer_column = pick_column(
        dataset.column_names,
        args.answer_column,
        ["answer", "answer_text", "reference_answer", "gold_answer", "open_answer", "label"],
        required=False,
    )

    print(f"Rows: {len(dataset)}")
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")
    if answer_column:
        print(f"Gold column: {answer_column}")

    model_name = args.adapter or args.model
    print(f"Loading model/adapter: {model_name}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    FastVisionModel.for_inference(model)

    predictions: List[Dict[str, str]] = []
    exact = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="Qwen3.6-Unsloth-OpenQA"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            image = select_image_variant(image, args.image_variant, args.enhance_longest_side)
            inputs = build_inputs(processor, image, prompt, args.answer_prefill).to("cuda")

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            raw_text = processor.decode(
                generated_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            answer = clean_answer(raw_text, args.max_answer_chars)
            if not answer:
                answer = "N/A"

            predictions.append({"question_id": question_id, args.answer_field: answer})
            raw_row = {"question_id": question_id, args.answer_field: answer, "raw_text": raw_text}
            if answer_column:
                gold = str(row[answer_column]).strip()
                raw_row["gold"] = gold
                if gold and gold.upper() != "HIDDEN":
                    scored += 1
                    exact += int(normalize_for_match(answer) == normalize_for_match(gold))
            raw_file.write(json.dumps(raw_row, ensure_ascii=False) + "\n")

    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw outputs: {raw_output_path}")
    if scored:
        print(f"Exact match: {exact / scored:.4f} ({exact}/{scored})")


if __name__ == "__main__":
    main()
