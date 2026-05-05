import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_MODEL = "microsoft/Phi-4-reasoning-vision-15B"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_PROMPT = """You are solving a visual multiple-choice exam question.

Read the image carefully, including all question text, answer options, diagrams, charts, tables, labels, formulas, and units.

Think internally if needed, but output only the final option letter: A, B, C, D, or E.
Do not output explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phi-4-Reasoning-Vision-15B on Visual MCQ datasets.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--output", default="outputs/visual_mcq_phi4_reasoning_vision_15b.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--fallback-answer", choices=sorted(ANSWER_KEYS), default="A")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    parser.add_argument("--image-token", default=DEFAULT_IMAGE_TOKEN)
    parser.add_argument(
        "--reasoning-mode",
        default="nothink",
        choices=["auto", "nothink", "think"],
        help="Append Phi reasoning control token after the assistant generation prompt.",
    )
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
    text_without_think = re.sub(r"<THINK>.*?</THINK>", " ", text, flags=re.DOTALL)
    text_without_think = text_without_think.replace("<NOTHINK>", " ")
    patterns = [
        r"<ANSWER>\s*[\(\[]?\s*([A-E])\b.*?</ANSWER>",
        r"(?:FINAL\s+ANSWER|FINAL|THE\s+ANSWER)\s*(?:IS|:|-)?\s*[\(\[]?\s*([A-E])\b",
        r"(?:ANSWER|OPTION|CHOICE|SOLUTION)\s*(?:IS|:|-)?\s*[\(\[]?\s*([A-E])\b",
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


def patch_siglip2_filter_decorator() -> None:
    """Compatibility shim for Phi-4 remote code on newer Transformers builds."""
    try:
        from transformers.models.siglip2 import image_processing_siglip2 as siglip2_ips
    except Exception:
        return

    if not hasattr(siglip2_ips, "ChannelDimension"):
        try:
            from transformers.image_utils import ChannelDimension

            siglip2_ips.ChannelDimension = ChannelDimension
        except Exception:
            pass

    if hasattr(siglip2_ips, "filter_out_non_signature_kwargs"):
        return

    def filter_out_non_signature_kwargs() -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    siglip2_ips.filter_out_non_signature_kwargs = filter_out_non_signature_kwargs


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    patch_siglip2_filter_decorator()
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "dtype": dtype_from_arg(args.torch_dtype),
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        print(
            "Warning: --load-in-4bit is ignored for Phi-4-reasoning-vision-15B because "
            "its custom loader casts the model after loading, which conflicts with bitsandbytes."
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resolve_adapter_path(args.adapter))
    return model.eval()


def checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def resolve_adapter_path(adapter: str) -> str:
    adapter_path = Path(adapter)
    if not adapter_path.exists():
        return adapter
    if (adapter_path / "adapter_config.json").exists():
        return str(adapter_path)

    checkpoints = [
        child
        for child in adapter_path.glob("checkpoint-*")
        if child.is_dir() and (child / "adapter_config.json").exists()
    ]
    if checkpoints:
        latest = sorted(checkpoints, key=checkpoint_sort_key)[-1]
        print(f"Warning: adapter_config.json not found in {adapter_path}; using {latest}")
        return str(latest)

    raise FileNotFoundError(
        f"Adapter path exists but has no adapter_config.json: {adapter_path}. "
        "Training likely crashed before saving the final LoRA adapter."
    )


def model_device(model: torch.nn.Module) -> torch.device:
    model_device_attr = getattr(model, "device", None)
    if isinstance(model_device_attr, torch.device):
        return model_device_attr
    return next(model.parameters()).device


def build_prompt(processor: AutoProcessor, image_token: str, prompt: str, reasoning_mode: str) -> str:
    messages = [{"role": "user", "content": f"{image_token}\n{prompt}"}]
    formatted = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if reasoning_mode == "think":
        formatted += "<think>"
    elif reasoning_mode == "nothink":
        formatted += "<nothink>"
    return formatted


def build_inputs(processor: AutoProcessor, image: Image.Image, formatted_prompt: str) -> Dict[str, torch.Tensor]:
    inputs = processor(
        text=formatted_prompt,
        images=[image],
        return_tensors="pt",
    )
    return dict(inputs)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Phi-4 vision inference will be very slow on CPU.")

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
    patch_siglip2_filter_decorator()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args)
    device = model_device(model)

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="Phi-4-Vision"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            image = select_image_variant(image, args.image_variant, args.enhance_longest_side)
            formatted_prompt = build_prompt(processor, args.image_token, prompt, args.reasoning_mode)
            inputs = move_batch_to_device(build_inputs(processor, image, formatted_prompt), device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            raw_text = processor.tokenizer.decode(
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
