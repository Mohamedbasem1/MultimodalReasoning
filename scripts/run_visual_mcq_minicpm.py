import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_MODEL = "openbmb/MiniCPM-V-4_5"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"
DEFAULT_PROMPT = """You are solving a visual multiple-choice exam question.

Read the image carefully, including all question text, answer options, diagrams, charts, tables, labels, formulas, and units.

Think internally if needed, but output only the final option letter: A, B, C, D, or E.
Do not output explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniCPM-V 4.5 on Visual MCQ datasets.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_mcq_minicpm_v45.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--fallback-answer", choices=sorted(ANSWER_KEYS), default="A")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Enable Transformers low-memory/meta loading. Leave off if MiniCPM loading hits all_tied_weights_keys errors.",
    )
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable MiniCPM deep thinking mode.")
    parser.add_argument("--stream", action="store_true", help="Use streaming chat output.")
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


def dtype_from_arg(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": dtype_from_arg(args.torch_dtype),
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        return AutoModel.from_pretrained(args.model, **kwargs).eval()

    model = AutoModel.from_pretrained(args.model, **kwargs).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def run_chat(
    model: torch.nn.Module,
    tokenizer: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
    stream: bool,
) -> str:
    messages = [{"role": "user", "content": [image, prompt]}]
    answer = model.chat(
        msgs=messages,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        stream=stream,
        max_new_tokens=max_new_tokens,
    )
    if isinstance(answer, str):
        return answer
    if isinstance(answer, Iterable):
        return "".join(str(chunk) for chunk in answer)
    return str(answer)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

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
    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="MiniCPM-V"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            image = select_image_variant(image, args.image_variant, args.enhance_longest_side)
            raw_text = run_chat(
                model=model,
                tokenizer=tokenizer,
                image=image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                stream=args.stream,
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
