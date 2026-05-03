import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
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
    parser.add_argument(
        "--image-variants",
        nargs="+",
        default=["original"],
        choices=["original", "enhanced"],
        help="Run votes over one or more image variants. Use 'original enhanced' for stronger OCR/detail voting.",
    )
    parser.add_argument(
        "--enhance-longest-side",
        type=int,
        default=1600,
        help="Longest-side resize target for the enhanced image variant.",
    )
    parser.add_argument(
        "--ocr-engine",
        default="none",
        choices=["none", "easyocr", "paddleocr", "deepseek"],
        help="Optional external OCR engine. OCR text is injected into every prompt.",
    )
    parser.add_argument(
        "--ocr-json",
        default=None,
        help="Optional JSON/JSONL file with precomputed records containing question_id and ocr_text.",
    )
    parser.add_argument(
        "--ocr-langs",
        nargs="+",
        default=["en"],
        help="OCR language codes. EasyOCR accepts multiple codes; PaddleOCR uses the first one.",
    )
    parser.add_argument(
        "--ocr-max-chars",
        type=int,
        default=1800,
        help="Maximum OCR text characters appended to each prompt.",
    )
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=0.20,
        help="Ignore OCR spans below this confidence when the engine returns confidence.",
    )
    parser.add_argument(
        "--ocr-on-enhanced",
        action="store_true",
        help="Run OCR on the enhanced image variant instead of the original image.",
    )
    parser.add_argument(
        "--ocr-cpu",
        action="store_true",
        help="Run OCR on CPU even when CUDA is available. Useful if OCR competes with the VLM for GPU memory.",
    )
    parser.add_argument("--deepseek-ocr-model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument(
        "--deepseek-ocr-attn-implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation for DeepSeek-OCR. Use flash_attention_2 only if flash-attn is installed.",
    )
    parser.add_argument("--deepseek-ocr-base-size", type=int, default=1024)
    parser.add_argument("--deepseek-ocr-image-size", type=int, default=640)
    parser.add_argument("--deepseek-ocr-output-dir", default="outputs/deepseek_ocr")
    parser.add_argument(
        "--deepseek-ocr-no-crop",
        action="store_true",
        help="Disable DeepSeek-OCR crop mode. Crop mode is usually better for dense document images.",
    )
    parser.add_argument(
        "--deepseek-ocr-test-compress",
        action="store_true",
        help="Enable DeepSeek-OCR test_compress mode from the model card example.",
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


def build_image_variants(
    image: Image.Image,
    requested_variants: Sequence[str],
    enhance_longest_side: int,
) -> List[Tuple[str, Image.Image]]:
    variants: List[Tuple[str, Image.Image]] = []
    for variant in requested_variants:
        if variant == "original":
            variants.append(("original", image))
        elif variant == "enhanced":
            variants.append(("enhanced", enhance_image(image, enhance_longest_side)))
        else:
            raise ValueError(f"Unsupported image variant: {variant}")
    return variants


def augment_prompt_with_ocr(prompt: str, ocr_text: str) -> str:
    if not ocr_text:
        return prompt
    return (
        f"{prompt}\n\n"
        "External OCR text extracted from the same image is provided below. "
        "Use it only as supporting evidence; if it conflicts with the image, trust the image.\n"
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

    records = []
    if ocr_path.suffix.lower() == ".jsonl":
        with ocr_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        data = json.loads(ocr_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records = data
        else:
            raise ValueError("OCR JSON must be a list or JSONL records.")

    lookup: Dict[str, str] = {}
    for record in records:
        question_id = str(record.get("question_id", ""))
        ocr_text = str(record.get("ocr_text", "")).strip()
        if question_id and ocr_text:
            lookup[question_id] = ocr_text[:max_chars]
    return lookup


class OcrRunner:
    def __init__(
        self,
        engine: str,
        langs: Sequence[str],
        min_confidence: float,
        max_chars: int,
        use_gpu: bool,
        deepseek_model: str = "deepseek-ai/DeepSeek-OCR",
        deepseek_attn_implementation: str = "sdpa",
        deepseek_base_size: int = 1024,
        deepseek_image_size: int = 640,
        deepseek_output_dir: str = "outputs/deepseek_ocr",
        deepseek_crop_mode: bool = True,
        deepseek_test_compress: bool = False,
    ) -> None:
        self.engine = engine
        self.langs = list(langs)
        self.min_confidence = min_confidence
        self.max_chars = max_chars
        self.reader: Any = None
        self.tokenizer: Any = None
        self.deepseek_base_size = deepseek_base_size
        self.deepseek_image_size = deepseek_image_size
        self.deepseek_output_dir = Path(deepseek_output_dir)
        self.deepseek_crop_mode = deepseek_crop_mode
        self.deepseek_test_compress = deepseek_test_compress
        self.use_gpu = use_gpu

        if engine == "none":
            return
        if engine == "easyocr":
            import easyocr

            self.reader = easyocr.Reader(self.langs, gpu=use_gpu)
            return
        if engine == "paddleocr":
            from paddleocr import PaddleOCR

            self.reader = PaddleOCR(use_angle_cls=True, lang=self.langs[0])
            return
        if engine == "deepseek":
            from transformers import AutoModel, AutoTokenizer

            self.deepseek_output_dir.mkdir(parents=True, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(deepseek_model, trust_remote_code=True)
            self.reader = AutoModel.from_pretrained(
                deepseek_model,
                _attn_implementation=deepseek_attn_implementation,
                trust_remote_code=True,
                use_safetensors=True,
            )
            self.reader = self.reader.eval()
            if use_gpu:
                self.reader = self.reader.cuda().to(torch.bfloat16)
            return
        raise ValueError(f"Unsupported OCR engine: {engine}")

    @property
    def enabled(self) -> bool:
        return self.engine != "none" and self.reader is not None

    def extract_text(self, image: Image.Image) -> str:
        if not self.enabled:
            return ""
        if self.engine == "easyocr":
            return self._extract_easyocr(image)
        if self.engine == "paddleocr":
            return self._extract_paddleocr(image)
        if self.engine == "deepseek":
            return self._extract_deepseek(image)
        return ""

    def _clip(self, text: str) -> str:
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars].rsplit("\n", 1)[0]

    def _extract_easyocr(self, image: Image.Image) -> str:
        import numpy as np

        results = self.reader.readtext(np.array(image), paragraph=False)
        lines = []
        for _bbox, text, confidence in results:
            if confidence is None or confidence >= self.min_confidence:
                lines.append(str(text))
        return self._clip("\n".join(lines))

    def _extract_paddleocr(self, image: Image.Image) -> str:
        import numpy as np

        results = self.reader.ocr(np.array(image), cls=True)
        lines = []
        for page in results or []:
            for item in page or []:
                if not item or len(item) < 2:
                    continue
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and text_info:
                    text = str(text_info[0])
                    confidence = float(text_info[1]) if len(text_info) > 1 else 1.0
                    if confidence >= self.min_confidence:
                        lines.append(text)
        return self._clip("\n".join(lines))

    def _extract_deepseek(self, image: Image.Image) -> str:
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as image_file:
            image.save(image_file.name, format="JPEG", quality=95)
            result = self.reader.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_file.name,
                output_path=str(self.deepseek_output_dir),
                base_size=self.deepseek_base_size,
                image_size=self.deepseek_image_size,
                crop_mode=self.deepseek_crop_mode,
                save_results=False,
                test_compress=self.deepseek_test_compress,
            )
        if result is None:
            return ""
        return self._clip(str(result))


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
    print(f"Image variants: {', '.join(args.image_variants)}")
    ocr_lookup = load_ocr_json(args.ocr_json, args.ocr_max_chars)
    if ocr_lookup:
        print(f"Loaded precomputed OCR records: {len(ocr_lookup)}")

    try:
        ocr_runner = OcrRunner(
            engine=args.ocr_engine,
            langs=args.ocr_langs,
            min_confidence=args.ocr_min_confidence,
            max_chars=args.ocr_max_chars,
            use_gpu=torch.cuda.is_available() and not args.ocr_cpu,
            deepseek_model=args.deepseek_ocr_model,
            deepseek_attn_implementation=args.deepseek_ocr_attn_implementation,
            deepseek_base_size=args.deepseek_ocr_base_size,
            deepseek_image_size=args.deepseek_ocr_image_size,
            deepseek_output_dir=args.deepseek_ocr_output_dir,
            deepseek_crop_mode=not args.deepseek_ocr_no_crop,
            deepseek_test_compress=args.deepseek_ocr_test_compress,
        )
    except Exception as exc:
        print(f"Warning: OCR engine '{args.ocr_engine}' could not be initialized: {exc}")
        print("Continuing without external OCR.")
        ocr_runner = OcrRunner("none", [], args.ocr_min_confidence, args.ocr_max_chars, False)
    if ocr_runner.enabled:
        print(f"OCR engine: {args.ocr_engine} langs={','.join(args.ocr_langs)}")

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
            image_variants = build_image_variants(
                image,
                requested_variants=args.image_variants,
                enhance_longest_side=args.enhance_longest_side,
            )
            ocr_image = image_variants[-1][1] if args.ocr_on_enhanced else image
            ocr_text = ocr_lookup.get(question_id, "")
            if not ocr_text:
                ocr_text = ocr_runner.extract_text(ocr_image)
            votes = []

            for variant_name, variant_image in image_variants:
                for prompt_name, prompt in prompts:
                    final_prompt = augment_prompt_with_ocr(prompt, ocr_text)
                    answer_key, raw_text = generate_one(
                        model=model,
                        processor=processor,
                        image=variant_image,
                        prompt=final_prompt,
                        max_new_tokens=args.max_new_tokens,
                        fallback_answer=args.fallback_answer,
                    )
                    votes.append(
                        {
                            "variant": variant_name,
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
                "ocr_text": ocr_text,
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
