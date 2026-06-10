import argparse
import os
import json
import random
import re
import builtins
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
from datasets import Dataset, load_dataset
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"
DEFAULT_PROMPT = """You are answering a visual open-ended exam question.

Read the image carefully, including all question text, diagrams, charts, tables, labels, formulas, and units.

Learn and follow the structure of the reference answers in training: concise wording, same language as the question when possible, correct units, exact numbers, and no unnecessary sentence framing.

Think internally if needed, but output only the concise final answer text.
Do not output explanation, reasoning, chain-of-thought, or <think> tags."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental Unsloth QLoRA for Qwen3.6-35B-A3B on Visual OpenQA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--output-dir", default="outputs/qwen36-35b-a3b-openqa-unsloth-lora")
    parser.add_argument("--prompt-file", default="prompts/visual_openqa_prompt.txt")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="auto")
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--include-language", nargs="+", default=None)
    parser.add_argument("--include-subject-contains", nargs="+", default=None)
    parser.add_argument("--weak-filter-mode", default="or", choices=["or", "and"])
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-from-train", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-answer-chars", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1000)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def clean_answer(value: Any, max_chars: int) -> Optional[str]:
    answer = str(value).strip()
    if re.search(r"</think>", answer, flags=re.IGNORECASE):
        answer = re.split(r"</think>", answer, flags=re.IGNORECASE)[-1]
    else:
        answer = re.sub(r"<think>.*", " ", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"</?think>", " ", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.strip(" \t\r\n\"'")
    if not answer or answer.lower() in {"none", "nan", "null", "hidden"}:
        return None
    if max_chars > 0 and len(answer) > max_chars:
        answer = answer[:max_chars].rstrip()
    return answer


def filter_matches(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    checks = []
    if args.include_language:
        checks.append("language" in row and str(row["language"]) in set(args.include_language))
    if args.include_subject_contains:
        subject = str(row.get("subject", "")).lower()
        checks.append(any(fragment.lower() in subject for fragment in args.include_subject_contains))
    if not checks:
        return True
    if args.weak_filter_mode == "and":
        return all(checks)
    return any(checks)


def filter_dataset(dataset: Dataset, args: argparse.Namespace, answer_column: str) -> Dataset:
    if args.filter_type:
        allowed_types = set(args.filter_type)
        dataset = dataset.filter(lambda row: row.get("type") in allowed_types)
    if args.include_language or args.include_subject_contains:
        dataset = dataset.filter(lambda row: filter_matches(row, args))
    dataset = dataset.filter(lambda row: clean_answer(row.get(answer_column), args.max_answer_chars) is not None)
    return dataset


def maybe_limit(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def convert_to_conversation(row: Dict[str, Any], prompt: str, image_column: str, answer_column: str, args: argparse.Namespace) -> Dict[str, Any]:
    image = normalize_image(row[image_column])
    image = select_image_variant(image, args.image_variant, args.enhance_longest_side)
    answer = clean_answer(row[answer_column], args.max_answer_chars)
    if answer is None:
        raise ValueError("Cannot convert row without an answer.")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(args.prompt_file)
    print(f"Loading train split: {args.dataset} [{args.train_split}]")
    dataset = load_dataset(args.dataset, split=args.train_split)
    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    image_column = pick_column(dataset.column_names, args.image_column, ["image", "image_id"])
    answer_column = pick_column(
        dataset.column_names,
        args.answer_column,
        ["answer", "answer_text", "reference_answer", "gold_answer", "open_answer", "label"],
    )
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")
    print(f"Answer column: {answer_column}")

    dataset = filter_dataset(dataset, args, answer_column)
    if args.validation_from_train > 0 and args.validation_from_train < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.validation_from_train, len(dataset)))
    dataset = maybe_limit(dataset, args.train_limit)
    print(f"Train rows: {len(dataset)}")

    print("Converting rows to Unsloth vision conversations")
    converted_rows = [
        convert_to_conversation(row, prompt, image_column, answer_column, args)
        for row in dataset
    ]

    training_args = vars(args)
    training_args.update(
        {
            "id_column": id_column,
            "image_column": image_column,
            "answer_column": answer_column,
            "train_rows": len(converted_rows),
        }
    )
    (output_dir / "training_args.json").write_text(json.dumps(training_args, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastVisionModel, is_bfloat16_supported
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as exc:
        raise ImportError(
            "Missing Unsloth stack. Install with: pip install -r requirements-unsloth-qwen36.txt"
        ) from exc

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )
    FastVisionModel.for_training(model)

    fp16 = not is_bfloat16_supported()
    bf16 = is_bfloat16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(
            model,
            processor,
            train_on_responses_only=False,
            completion_only_loss=True,
        ),
        train_dataset=converted_rows,
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            fp16=fp16,
            bf16=bf16,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_strategy="steps",
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_length,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(output_dir)
    print(f"Saved final LoRA adapter: {output_dir}")


if __name__ == "__main__":
    main()
