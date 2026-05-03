import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "deepseek-ai/DeepSeek-OCR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OCR/Markdown with DeepSeek-OCR without loading the Qwen answer model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="MBZUAI/EXAMS-V")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/deepseek_ocr.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument(
        "--filter-type",
        nargs="+",
        default=None,
        help="Optional values for the dataset 'type' column, e.g. image image_text.",
    )
    parser.add_argument(
        "--image-variant",
        default="enhanced",
        choices=["original", "enhanced"],
        help="Image variant sent to DeepSeek-OCR.",
    )
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    parser.add_argument("--ocr-max-chars", type=int, default=3000)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output-dir", default="outputs/deepseek_ocr_cache")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Use eager for widest compatibility; flash_attention_2 only if flash-attn is installed.",
    )
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--test-compress", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


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


def prepare_image(image: Image.Image, variant: str, longest_side: int) -> Image.Image:
    if variant == "original":
        return image
    if variant == "enhanced":
        return enhance_image(image, longest_side)
    raise ValueError(f"Unsupported image variant: {variant}")


def clip_text(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    text = "\n".join(line.strip() for line in str(text).splitlines() if line.strip())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0]


def extract_ocr(
    model: Any,
    tokenizer: Any,
    image: Image.Image,
    output_dir: Path,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    test_compress: bool,
    max_chars: int,
) -> str:
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as image_file:
        image.save(image_file.name, format="JPEG", quality=95)
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file.name,
            output_path=str(output_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            test_compress=test_compress,
        )
    return clip_text(result, max_chars)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DeepSeek-OCR: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        _attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval()
    if torch.cuda.is_available() and not args.cpu:
        model = model.cuda().to(torch.bfloat16)

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.filter_type:
        allowed_types = set(args.filter_type)
        dataset = dataset.filter(lambda row: row.get("type") in allowed_types)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    image_column = pick_column(dataset.column_names, args.image_column, ["image", "image_id"])
    print(f"Rows: {len(dataset)}")
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")

    with output_path.open("w", encoding="utf-8") as output_file:
        for row in tqdm(dataset, desc="DeepSeek OCR"):
            question_id = str(row[id_column])
            image = normalize_image(row[image_column])
            image = prepare_image(image, args.image_variant, args.enhance_longest_side)
            try:
                ocr_text = extract_ocr(
                    model=model,
                    tokenizer=tokenizer,
                    image=image,
                    output_dir=output_dir,
                    base_size=args.base_size,
                    image_size=args.image_size,
                    crop_mode=not args.no_crop,
                    test_compress=args.test_compress,
                    max_chars=args.ocr_max_chars,
                )
                error = None
            except Exception as exc:
                ocr_text = ""
                error = str(exc)

            record = {
                "question_id": question_id,
                "ocr_text": ocr_text,
            }
            if error:
                record["error"] = error
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote OCR JSONL: {output_path}")


if __name__ == "__main__":
    main()

