import argparse
import builtins
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


os.environ.setdefault("UNSLOTH_MOE_BACKEND", "native_torch")


def patch_unsloth_transformers_symbols() -> None:
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


ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"
DEFAULT_PROMPT = """Solve the visual multiple-choice exam question in the image.

Read all question text, answer options, diagrams, charts, tables, formulas, labels, and units.

Choose exactly one correct option.

Output only one uppercase letter: A, B, C, D, or E.
Do not output explanation, analysis, markdown, or <think> tags."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3.6-35B-A3B Unsloth on Visual MCQ.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="Optional Unsloth/PEFT LoRA adapter directory.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_qwen36_unsloth.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--fallback-answer", choices=sorted(ANSWER_KEYS), default="A")
    parser.add_argument(
        "--selection-method",
        choices=["logits", "generate"],
        default="logits",
        help="Use next-token option scoring by default; generate keeps the older free-text parser path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--answer-prefill", default="ANSWER: ")
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


def pick_column(columns: Sequence[str], requested: str, candidates: Iterable[str]) -> str:
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"Column '{requested}' not found. Available: {list(columns)}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not infer column. Available: {list(columns)}")


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


def parse_answer(raw_text: str, fallback: str) -> str:
    text = raw_text.strip().upper()
    if re.search(r"</THINK>", text):
        parts = [part.strip() for part in re.split(r"</THINK>", text) if part.strip()]
        text = max(parts, key=len) if parts else text
    text = re.sub(r"</?THINK>", " ", text)
    patterns = [
        r"<ANSWER>\s*[\(\[]?\s*([A-E])\b.*?</ANSWER>",
        r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|-)?\s*[\(\[]?\s*([A-E])\b",
        r"^[\s\(\[]*([A-E])[\s\)\].,:;-]*$",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return fallback


def normalize_for_match(text: Any) -> str:
    value = str(text).upper().strip()
    return value if value in ANSWER_KEYS else ""


def option_token_ids(processor: Any) -> Dict[str, List[int]]:
    tokenizer = getattr(processor, "tokenizer", processor)
    result: Dict[str, List[int]] = {}
    for key in sorted(ANSWER_KEYS):
        ids = set()
        for text in (key, f" {key}", f"{key}.", f"{key})", f"({key})"):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.add(int(token_ids[0]))
        if not ids:
            token_ids = tokenizer.encode(key, add_special_tokens=False)
            if token_ids:
                ids.add(int(token_ids[0]))
        result[key] = sorted(ids)
    return result


def score_answer_logits(model: torch.nn.Module, inputs: Dict[str, torch.Tensor], token_ids_by_answer: Dict[str, List[int]], fallback: str) -> tuple[str, Dict[str, float]]:
    outputs = model(**inputs, use_cache=False, return_dict=True)
    logits = outputs.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    scores: Dict[str, float] = {}
    for answer, token_ids in token_ids_by_answer.items():
        valid_ids = [token_id for token_id in token_ids if token_id < log_probs.numel()]
        if valid_ids:
            scores[answer] = float(log_probs[valid_ids].max().item())
    if not scores:
        return fallback, scores
    return max(scores, key=scores.get), scores


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
        raise ImportError("Missing Unsloth stack. Install with: pip install -r requirements-unsloth-qwen36.txt") from exc

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
    has_gold = args.answer_column in dataset.column_names

    print(f"Rows: {len(dataset)}")
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")
    if has_gold:
        print(f"Gold column: {args.answer_column}")

    model_name = args.adapter or args.model
    print(f"Loading model/adapter: {model_name}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    FastVisionModel.for_inference(model)
    token_ids_by_answer = option_token_ids(processor)
    if args.selection_method == "logits":
        print(f"Using next-token MCQ scoring with option token IDs: {token_ids_by_answer}")

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="Qwen3.6-Unsloth-MCQ"):
            question_id = str(row[id_column])
            image = select_image_variant(normalize_image(row[image_column]), args.image_variant, args.enhance_longest_side)
            inputs = build_inputs(processor, image, prompt, args.answer_prefill).to("cuda")

            with torch.inference_mode():
                if args.selection_method == "logits":
                    answer_key, scores = score_answer_logits(model, inputs, token_ids_by_answer, args.fallback_answer)
                    raw_text = ""
                else:
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
                    answer_key = parse_answer(raw_text, args.fallback_answer)
                    scores = {}
            predictions.append({"question_id": question_id, "answer_key": answer_key})
            raw_row = {"question_id": question_id, "answer_key": answer_key, "raw_text": raw_text}
            if scores:
                raw_row["scores"] = scores

            if has_gold:
                gold = normalize_for_match(row[args.answer_column])
                raw_row["gold"] = gold or str(row[args.answer_column])
                if gold:
                    scored += 1
                    correct += int(answer_key == gold)
            raw_file.write(json.dumps(raw_row, ensure_ascii=False) + "\n")

    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw outputs: {raw_output_path}")
    if scored:
        print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")


if __name__ == "__main__":
    main()
