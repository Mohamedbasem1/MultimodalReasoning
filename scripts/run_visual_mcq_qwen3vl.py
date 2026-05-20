import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - checked at runtime on Lightning.
    raise ImportError(
        "Qwen3VLForConditionalGeneration is missing. Install the Qwen3 requirements first: "
        "pip install -r requirements-qwen3vl.txt"
    ) from exc


ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"
DEFAULT_PROMPT = """You are solving a visual multiple-choice exam question.

Read the image carefully, including diagrams, charts, equations, labels, units, tables, and all answer options.

Think internally if needed, but output only the final option letter: A, B, C, D, or E.
Do not output explanation or chain-of-thought."""
TOKENIZED_CHAT_PROCESSOR_KWARGS = {"return_tensors": "pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-8B-Thinking on Visual MCQ datasets.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT/LoRA adapter directory produced by train_visual_mcq_lora_qwen3vl.py.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_qwen3vl8b_thinking.json")
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
    parser.add_argument("--selection-method", choices=["logits", "generate"], default="generate")
    parser.add_argument("--answer-prefill", default="\nAnswer: ")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=[None, "flash_attention_2", "sdpa", "eager"],
        help="Use flash_attention_2 only if it is installed; otherwise leave unset or use sdpa.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading if available.",
    )
    parser.add_argument(
        "--image-variant",
        default="enhanced",
        choices=["original", "enhanced"],
        help="Image variant sent to Qwen3-VL.",
    )
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    parser.add_argument(
        "--ocr-json",
        default=None,
        help="Optional JSON/JSONL OCR records. Supports question_id/id/sample_id and ocr_text/text fields.",
    )
    parser.add_argument("--ocr-max-chars", type=int, default=2200)
    return parser.parse_args()


def load_prompt(path: str) -> str:
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def augment_prompt_with_ocr(prompt: str, ocr_text: str) -> str:
    ocr_text = ocr_text.strip()
    if not ocr_text:
        return prompt
    return (
        f"{prompt}\n\n"
        "External OCR text extracted from the same image is provided below. "
        "Use it only as supporting evidence. If OCR conflicts with the image, trust the image.\n"
        "<ocr_text>\n"
        f"{ocr_text}\n"
        "</ocr_text>"
    )


def load_ocr_json(path: Optional[str], max_chars: int) -> Dict[str, str]:
    if not path:
        return {}
    ocr_path = Path(path)
    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR JSON file not found: {path}")

    if ocr_path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with ocr_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    else:
        data = json.loads(ocr_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("OCR JSON must be a list or JSONL records.")
        records = data

    lookup: Dict[str, str] = {}
    for record in records:
        ocr_text = str(
            record.get("ocr_text")
            or record.get("text")
            or record.get("parsed_text")
            or record.get("ParsedText")
            or ""
        ).strip()
        if not ocr_text:
            continue
        for key in ("question_id", "id", "sample_id", "index"):
            value = record.get(key)
            if value is not None and str(value):
                lookup[str(value)] = ocr_text[:max_chars]
    return lookup


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
    text_without_think = re.sub(r"<THINK>.*?</THINK>", " ", text, flags=re.DOTALL)
    patterns = [
        r"<ANSWER>\s*[\(\[]?\s*([A-E])\b.*?</ANSWER>",
        r"(?:FINAL\s+ANSWER|FINAL|THE\s+ANSWER)\s*(?:IS|:|-)?\s*[\(\[]?\s*([A-E])\b",
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|-)?\s*[\(\[]?\s*([A-E])\b",
        r"^[\s\(\[]*([A-E])[\s\)\].,:;-]*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_without_think)
        if match:
            return match.group(1)
    candidates = re.findall(r"\b([A-E])\b", text_without_think)
    if candidates:
        return candidates[-1]
    candidates = re.findall(r"\b([A-E])\b", text)
    return candidates[-1] if candidates else fallback


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


def qwen3_from_pretrained(model_name: str, kwargs: Dict[str, Any]) -> Qwen3VLForConditionalGeneration:
    try:
        return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc) or "dtype" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **retry_kwargs)


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs: Dict[str, Any] = {
        "dtype": dtype_from_arg(args.torch_dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = qwen3_from_pretrained(args.model, kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    return model.eval()


def model_device(model: torch.nn.Module) -> torch.device:
    model_device_attr = getattr(model, "device", None)
    if isinstance(model_device_attr, torch.device):
        return model_device_attr
    return next(model.parameters()).device


def build_inputs(processor: AutoProcessor, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
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
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs=TOKENIZED_CHAT_PROCESSOR_KWARGS,
    )
    return dict(inputs)


def apply_chat_template_text(processor: AutoProcessor, image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_score_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    answer_prefill: str,
) -> Dict[str, torch.Tensor]:
    text = apply_chat_template_text(processor, image, prompt) + answer_prefill
    try:
        return dict(processor(text=[text], images=[image], return_tensors="pt"))
    except TypeError:
        return dict(processor(text=text, images=image, return_tensors="pt"))


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def option_token_ids(processor: AutoProcessor) -> Dict[str, List[int]]:
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


def score_answer_logits(
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    token_ids_by_answer: Dict[str, List[int]],
    fallback: str,
) -> tuple[str, Dict[str, float]]:
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


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Qwen3-VL-8B inference will be very slow on CPU.")

    prompt = load_prompt(args.prompt_file)
    ocr_lookup = load_ocr_json(args.ocr_json, args.ocr_max_chars)
    if ocr_lookup:
        print(f"Loaded OCR text for {len(ocr_lookup)} ids from {args.ocr_json}")
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
    device = model_device(model)
    token_ids_by_answer = option_token_ids(processor)
    if args.selection_method == "logits":
        print(f"Using next-token MCQ scoring with option token IDs: {token_ids_by_answer}")

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for index, row in enumerate(tqdm(dataset, desc="Qwen3-VL")):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            image = select_image_variant(image, args.image_variant, args.enhance_longest_side)
            ocr_text = ocr_lookup.get(question_id, ocr_lookup.get(str(index), ""))
            row_prompt = augment_prompt_with_ocr(prompt, ocr_text)

            with torch.inference_mode():
                if args.selection_method == "logits":
                    inputs = move_batch_to_device(
                        build_score_inputs(processor, image, row_prompt, args.answer_prefill),
                        device,
                    )
                    answer_key, scores = score_answer_logits(
                        model,
                        inputs,
                        token_ids_by_answer,
                        args.fallback_answer,
                    )
                    raw_text = ""
                else:
                    inputs = move_batch_to_device(build_inputs(processor, image, row_prompt), device)
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )

                    trimmed_ids = [
                        output_ids[len(input_ids) :]
                        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
                    ]
                    raw_text = processor.batch_decode(
                        trimmed_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    answer_key = parse_answer(raw_text, args.fallback_answer)
                    scores = {}

            predictions.append({"question_id": question_id, "answer_key": answer_key})
            raw_row = {"question_id": question_id, "answer_key": answer_key, "raw_text": raw_text}
            if ocr_text:
                raw_row["ocr_text"] = ocr_text
            if scores:
                raw_row["scores"] = scores
            raw_file.write(json.dumps(raw_row, ensure_ascii=False) + "\n")

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
