import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"
DEFAULT_PROMPT = """You are answering a visual open-ended exam question.

Read the image carefully, including all question text, diagrams, charts, tables, labels, formulas, and units.

Learn and follow the structure of the reference answers in training: concise wording, same language as the question when possible, correct units, exact numbers, and no unnecessary sentence framing.

Think internally if needed, but output only the concise final answer text.
Do not output explanation, reasoning, chain-of-thought, or <think> tags.
/no_think"""
TOKENIZED_CHAT_PROCESSOR_KWARGS = {"return_tensors": "pt"}
NO_THINK_PREFILL = "</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-8B-Thinking on ImageCLEF Visual OpenQA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_openqa_qwen3vl8b_thinking.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--gold-output", default=None, help="Optional gold file matching the selected rows.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-file", default="prompts/visual_openqa_prompt.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="auto", help="Optional gold answer column for labeled splits.")
    parser.add_argument("--answer-field", default="answer", help="Submission answer field name.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-answer-chars", type=int, default=1000)
    parser.add_argument("--enable-thinking", action="store_true", help="Allow Qwen3 thinking mode if supported.")
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
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
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    else:
        text = re.sub(r"<think>.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<answer>|</answer>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:final\s+answer|answer)\s*(?:is|:|-)?\s*", "", text, flags=re.IGNORECASE)
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


def write_json_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_existing_predictions(output_path: Path, raw_output_path: Path, answer_field: str) -> List[Dict[str, str]]:
    rows = read_json_rows(output_path)
    if not rows:
        rows = read_jsonl_rows(raw_output_path)

    seen: set[str] = set()
    predictions: List[Dict[str, str]] = []
    for row in rows:
        question_id = row.get("question_id")
        answer = row.get(answer_field)
        if question_id is None or answer is None:
            continue
        question_id_text = str(question_id)
        if question_id_text in seen:
            continue
        seen.add(question_id_text)
        predictions.append({"question_id": question_id_text, answer_field: str(answer)})
    return predictions


def export_gold(
    path: Path,
    dataset: Any,
    id_column: str,
    answer_column: str,
    answer_field: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    for row in dataset:
        answer = str(row.get(answer_column, "")).strip()
        if answer_field == "answers":
            rows.append({"question_id": str(row[id_column]), "answers": [answer]})
        else:
            rows.append({"question_id": str(row[id_column]), "answers": [answer]})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, rows)


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
    def load_with_dtype_retry(load_kwargs: Dict[str, Any]) -> Qwen3VLForConditionalGeneration:
        try:
            return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc) or "dtype" not in load_kwargs:
                raise
            retry_kwargs = dict(load_kwargs)
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
            return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **retry_kwargs)

    try:
        return load_with_dtype_retry(kwargs)
    except NotImplementedError as exc:
        if "Cannot copy out of meta tensor" not in str(exc):
            raise
        if kwargs.get("device_map") != "auto" or not torch.cuda.is_available():
            raise
        print("device_map=auto hit a meta tensor dispatch issue; retrying with device_map='cuda'.")
        retry_kwargs = dict(kwargs)
        retry_kwargs["device_map"] = "cuda"
        return load_with_dtype_retry(retry_kwargs)


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


def build_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    enable_thinking: bool,
) -> Dict[str, torch.Tensor]:
    if not enable_thinking:
        no_think_inputs = build_no_think_prefill_inputs(processor, image, prompt)
        if no_think_inputs is not None:
            return no_think_inputs

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = apply_chat_template(
        processor=processor,
        messages=messages,
        enable_thinking=enable_thinking,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs=TOKENIZED_CHAT_PROCESSOR_KWARGS,
    )
    return ensure_tensor_inputs(dict(inputs))


def build_no_think_prefill_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
) -> Dict[str, torch.Tensor] | None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": NO_THINK_PREFILL}]},
    ]
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            processor_kwargs=TOKENIZED_CHAT_PROCESSOR_KWARGS,
        )
    except (TypeError, ValueError):
        return None
    return ensure_tensor_inputs(dict(inputs))


def apply_chat_template(
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    enable_thinking: bool,
    **kwargs: Any,
) -> Any:
    if enable_thinking:
        try:
            return processor.apply_chat_template(messages, enable_thinking=True, **kwargs)
        except TypeError:
            pass
    return processor.apply_chat_template(messages, **kwargs)


def tensorize_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, list):
        if not value:
            return value
        if all(torch.is_tensor(item) for item in value):
            return torch.stack(value)
        try:
            return torch.tensor(value)
        except (TypeError, ValueError):
            return value
    return value


def ensure_tensor_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    tensor_inputs = {key: tensorize_value(value) for key, value in inputs.items()}
    for key in ("input_ids", "attention_mask"):
        value = tensor_inputs.get(key)
        if torch.is_tensor(value) and value.ndim == 1:
            tensor_inputs[key] = value.unsqueeze(0)
    return tensor_inputs


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Qwen3-VL-8B inference will be very slow on CPU.")

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
    if args.gold_output and answer_column:
        export_gold(Path(args.gold_output), dataset, id_column, answer_column, args.answer_field)
        print(f"Wrote gold file: {args.gold_output}")

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = load_model(args)
    device = model_device(model)

    predictions: List[Dict[str, str]] = []
    completed_ids: set[str] = set()
    if args.resume:
        predictions = load_existing_predictions(output_path, raw_output_path, args.answer_field)
        completed_ids = {row["question_id"] for row in predictions}
        if completed_ids:
            print(f"Resuming with {len(completed_ids)} completed prediction(s).")
    exact = 0
    scored = 0

    raw_mode = "a" if args.resume and raw_output_path.exists() else "w"
    with raw_output_path.open(raw_mode, encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="Qwen3-OpenQA"):
            question_id = str(row[id_column])
            if question_id in completed_ids:
                continue
            image = normalize_image(row[image_column])
            image = select_image_variant(image, args.image_variant, args.enhance_longest_side)
            inputs = move_batch_to_device(build_inputs(processor, image, prompt, args.enable_thinking), device)

            with torch.inference_mode():
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
            raw_file.flush()
            completed_ids.add(question_id)
            write_json_atomic(output_path, predictions)

    write_json_atomic(output_path, predictions)
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw outputs: {raw_output_path}")
    if scored:
        print(f"Exact match: {exact / scored:.4f} ({exact}/{scored})")


if __name__ == "__main__":
    main()
