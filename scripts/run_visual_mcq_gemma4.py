import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor


ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"
DEFAULT_PROMPT = """You are solving a visual multiple-choice exam question.

Read the image carefully, including question text, answer options, diagrams, charts, equations, labels, units, and tables.

Choose exactly one correct option.

Output only one uppercase letter: A, B, C, D, or E.
Do not output reasoning, explanation, markdown, or XML tags."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma 4 vision model on Visual MCQ datasets.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_gemma4_31b.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--fallback-answer", choices=sorted(ANSWER_KEYS), default="A")
    parser.add_argument("--selection-method", choices=["logits", "generate"], default="logits")
    parser.add_argument("--answer-prefill", default="\nAnswer: ")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None, choices=[None, "sdpa", "eager", "flash_attention_2"])
    parser.add_argument(
        "--model-loader",
        choices=["auto", "image-text-to-text", "causal-lm"],
        default="auto",
        help="ERNIE 4.5 VL currently needs causal-lm despite being an image-text-to-text model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/processor code. Required for some newer VLMs such as ERNIE 4.5 VL.",
    )
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


def model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "dtype": dtype_from_arg(args.torch_dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def from_pretrained_with_dtype_retry(model_cls: Any, model_name: str, kwargs: Dict[str, Any]) -> torch.nn.Module:
    try:
        return model_cls.from_pretrained(model_name, **kwargs).eval()
    except TypeError as exc:
        if "dtype" not in str(exc) or "dtype" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        return model_cls.from_pretrained(model_name, **retry_kwargs).eval()


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs = model_kwargs(args)
    if args.model_loader == "causal-lm":
        from transformers import AutoModelForCausalLM

        return from_pretrained_with_dtype_retry(AutoModelForCausalLM, args.model, kwargs)

    try:
        from transformers import AutoModelForImageTextToText
        image_model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForMultimodalLM
        image_model_cls = AutoModelForMultimodalLM

    try:
        return from_pretrained_with_dtype_retry(image_model_cls, args.model, kwargs)
    except ValueError as exc:
        if args.model_loader != "auto":
            raise
        print(f"Warning: image-text loader failed, retrying with AutoModelForCausalLM: {exc}")
        from transformers import AutoModelForCausalLM

        return from_pretrained_with_dtype_retry(AutoModelForCausalLM, args.model, kwargs)


def repair_meta_rotary_tensors(model: torch.nn.Module) -> int:
    repaired = 0
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if not torch.is_tensor(inv_freq) or inv_freq.device.type != "meta":
            continue
        dim = int(inv_freq.numel()) * 2
        if dim <= 0:
            continue
        theta = float(getattr(module, "theta", getattr(module, "base", 10000.0)))
        module.inv_freq = 1.0 / theta ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )
        repaired += 1
    return repaired


def model_device(model: torch.nn.Module) -> torch.device:
    model_device_attr = getattr(model, "device", None)
    if isinstance(model_device_attr, torch.device) and model_device_attr.type != "meta":
        return model_device_attr
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    for buffer in model.buffers():
        if buffer.device.type != "meta":
            return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def parse_answer(raw_text: str, fallback: str) -> str:
    text = raw_text.strip().upper()
    text = re.sub(r"<\|CHANNEL\>THOUGHT.*?<CHANNEL\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    patterns = [
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


def score_answer_logits(
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    token_ids_by_answer: Dict[str, List[int]],
    fallback: str,
) -> Tuple[str, Dict[str, float]]:
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


def apply_chat_template_text(processor: Any, messages: List[Dict[str, Any]]) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            pass
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_ernie_messages(image: Image.Image, prompt: str, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    if image_path:
        image_content = {"type": "image_url", "image_url": {"url": image_path}}
    else:
        image_content = {"type": "image", "image": image}
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content,
            ],
        }
    ]


def processor_call_with_vision(processor: Any, messages: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
    process_vision_info = getattr(processor, "process_vision_info", None)
    if not callable(process_vision_info):
        return None
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def build_inputs(processor: Any, image: Image.Image, prompt: str, answer_prefill: str, image_path: Optional[str] = None) -> Dict[str, Any]:
    messages = build_ernie_messages(image, prompt, image_path=image_path)
    text = apply_chat_template_text(processor, messages) + answer_prefill
    vision_inputs = processor_call_with_vision(processor, messages, text)
    if vision_inputs is not None:
        return vision_inputs
    try:
        return processor(text=[text], images=[image], return_tensors="pt")
    except TypeError:
        return processor(text=text, images=image, return_tensors="pt")


def build_generate_inputs(processor: Any, image: Image.Image, prompt: str, image_path: Optional[str] = None) -> Dict[str, Any]:
    messages = build_ernie_messages(image, prompt, image_path=image_path)
    text = apply_chat_template_text(processor, messages)
    vision_inputs = processor_call_with_vision(processor, messages, text)
    if vision_inputs is not None:
        return vision_inputs
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )


def decode_response(processor: Any, output_ids: torch.Tensor, input_len: int) -> str:
    response = processor.decode(output_ids[input_len:], skip_special_tokens=False)
    parse_response = getattr(processor, "parse_response", None)
    if callable(parse_response):
        try:
            parsed = parse_response(response)
            if isinstance(parsed, str):
                return parsed
            if isinstance(parsed, dict):
                for key in ("answer", "content", "text", "response"):
                    if key in parsed:
                        return str(parsed[key])
        except Exception:
            pass
    return response


def save_temp_image_for_processor(image: Image.Image, processor: Any) -> Optional[str]:
    if not callable(getattr(processor, "process_vision_info", None)):
        return None
    handle = tempfile.NamedTemporaryFile(prefix="ernie_vl_", suffix=".png", delete=False)
    temp_path = handle.name
    handle.close()
    image.save(temp_path)
    return temp_path


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Gemma 4 31B inference will be very slow on CPU.")

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

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args)
    repaired_rotary_tensors = repair_meta_rotary_tensors(model)
    if repaired_rotary_tensors:
        print(f"Repaired {repaired_rotary_tensors} meta rotary tensor(s).")
    add_image_preprocess = getattr(model, "add_image_preprocess", None)
    if callable(add_image_preprocess):
        add_image_preprocess(processor)
    device = model_device(model)
    token_ids_by_answer = option_token_ids(processor)
    if args.selection_method == "logits":
        print(f"Using next-token MCQ scoring with option token IDs: {token_ids_by_answer}")

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="Gemma4-MCQ"):
            question_id = str(row[id_column])
            image = select_image_variant(normalize_image(row[image_column]), args.image_variant, args.enhance_longest_side)
            temp_image_path = save_temp_image_for_processor(image, processor)

            try:
                with torch.inference_mode():
                    if args.selection_method == "logits":
                        inputs = move_batch_to_device(
                            build_inputs(processor, image, prompt, args.answer_prefill, image_path=temp_image_path),
                            device,
                        )
                        answer_key, scores = score_answer_logits(model, inputs, token_ids_by_answer, args.fallback_answer)
                        raw_text = ""
                    else:
                        inputs = move_batch_to_device(
                            build_generate_inputs(processor, image, prompt, image_path=temp_image_path),
                            device,
                        )
                        input_len = inputs["input_ids"].shape[-1]
                        outputs = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                        )
                        raw_text = decode_response(processor, outputs[0], input_len)
                        answer_key = parse_answer(raw_text, args.fallback_answer)
                        scores = {}
            finally:
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

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
