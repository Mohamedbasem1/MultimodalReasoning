import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_cosine_schedule_with_warmup

from run_visual_mcq_phi4vision import (
    DEFAULT_IMAGE_TOKEN,
    patch_siglip2_filter_decorator,
    parse_answer,
    select_image_variant,
)


ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_MODEL = "microsoft/Phi-4-reasoning-vision-15B"
DEFAULT_DATASET = "MBZUAI/EXAMS-V"
DEFAULT_PROMPT = """You are solving a visual multiple-choice exam question.

Read the image carefully, including all question text, answer options, diagrams, charts, tables, labels, formulas, and units.

Think internally if needed, but output only the final option letter: A, B, C, D, or E.
Do not output explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Phi-4-Reasoning-Vision-15B on Visual MCQ data.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--init-adapter", default=None, help="Optional existing PEFT/LoRA adapter to continue training from.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--output-dir", default="outputs/phi4-reasoning-vision-15b-examsv-lora")
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--include-language", nargs="+", default=None)
    parser.add_argument("--include-grade", nargs="+", default=None)
    parser.add_argument("--include-binary-columns", nargs="+", default=None)
    parser.add_argument("--include-subject-contains", nargs="+", default=None)
    parser.add_argument("--weak-filter-mode", default="or", choices=["or", "and"])
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        help="Phi/Phi3-style modules. If PEFT says these are missing, inspect model.named_modules().",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100, help="Set to 0 to disable generation eval during training.")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-generate-limit", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Accepted for command compatibility, but ignored for Phi.")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
    parser.add_argument("--reasoning-mode", default="nothink", choices=["auto", "nothink", "think"])
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


def normalize_answer(value: Any) -> Optional[str]:
    answer = str(value).strip().upper()
    match = re.search(r"[A-E]", answer)
    return match.group(0) if match else None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def weak_filter_matches(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    checks = []
    if args.include_language:
        checks.append("language" in row and str(row["language"]) in set(args.include_language))
    if args.include_grade:
        checks.append("grade" in row and str(row["grade"]) in set(str(grade) for grade in args.include_grade))
    if args.include_binary_columns:
        checks.append(any(column in row and truthy(row[column]) for column in args.include_binary_columns))
    if args.include_subject_contains:
        subject = str(row.get("subject", "")).lower()
        checks.append(any(fragment.lower() in subject for fragment in args.include_subject_contains))
    if not checks:
        return True
    if args.weak_filter_mode == "and":
        return all(checks)
    return any(checks)


def filter_dataset(dataset: Dataset, args: argparse.Namespace) -> Dataset:
    if args.filter_type:
        allowed_types = set(args.filter_type)
        dataset = dataset.filter(lambda row: row.get("type") in allowed_types)
    has_weak_filters = any(
        [
            args.include_language,
            args.include_grade,
            args.include_binary_columns,
            args.include_subject_contains,
        ]
    )
    if has_weak_filters:
        dataset = dataset.filter(lambda row: weak_filter_matches(row, args))
    dataset = dataset.filter(lambda row: normalize_answer(row.get(args.answer_column)) in ANSWER_KEYS)
    return dataset


def maybe_limit(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


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


def reasoning_suffix(mode: str) -> str:
    if mode == "think":
        return "<think>"
    if mode == "nothink":
        return "<nothink>"
    return ""


def build_formatted_prompt(processor: Any, prompt: str, answer: Optional[str], reasoning_mode: str) -> str:
    messages = [{"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"}]
    formatted = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    formatted += reasoning_suffix(reasoning_mode)
    if answer is not None:
        formatted += answer
    return formatted


def process_inputs(
    processor: Any,
    image: Image.Image,
    formatted_prompt: str,
    max_length: int,
    padding: bool,
) -> Dict[str, torch.Tensor]:
    inputs = processor(
        text=formatted_prompt,
        images=[image],
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return dict(inputs)


class VisualMcqCollator:
    def __init__(
        self,
        processor: Any,
        prompt: str,
        image_column: str,
        answer_column: str,
        max_length: int,
        image_variant: str,
        enhance_longest_side: int,
        reasoning_mode: str,
    ):
        self.processor = processor
        self.prompt = prompt
        self.image_column = image_column
        self.answer_column = answer_column
        self.max_length = max_length
        self.image_variant = image_variant
        self.enhance_longest_side = enhance_longest_side
        self.reasoning_mode = reasoning_mode

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = []
        prompt_texts = []
        images = []

        for row in rows:
            image = normalize_image(row[self.image_column])
            image = select_image_variant(image, self.image_variant, self.enhance_longest_side)
            answer = normalize_answer(row[self.answer_column])
            if answer is None:
                raise ValueError(f"Invalid answer: {row.get(self.answer_column)!r}")
            full_texts.append(build_formatted_prompt(self.processor, self.prompt, answer, self.reasoning_mode))
            prompt_texts.append(build_formatted_prompt(self.processor, self.prompt, None, self.reasoning_mode))
            images.append(image)

        full_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        full_inputs = dict(full_inputs)
        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[index, : int(prompt_length)] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[full_inputs["input_ids"] == pad_token_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


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
        print("Warning: --load-in-4bit is ignored for Phi-4 vision training.")

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def model_device(model: torch.nn.Module) -> torch.device:
    model_device_attr = getattr(model, "device", None)
    if isinstance(model_device_attr, torch.device):
        return model_device_attr
    return next(model.parameters()).device


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate_generation(
    model: torch.nn.Module,
    processor: Any,
    dataset: Dataset,
    prompt: str,
    image_column: str,
    answer_column: str,
    limit: int,
    max_new_tokens: int,
    image_variant: str,
    enhance_longest_side: int,
    reasoning_mode: str,
    max_length: int,
) -> Dict[str, float]:
    model.eval()
    total = min(limit, len(dataset))
    correct = 0

    for row in tqdm(dataset.select(range(total)), desc="Eval generation", leave=False):
        image = normalize_image(row[image_column])
        image = select_image_variant(image, image_variant, enhance_longest_side)
        expected = normalize_answer(row[answer_column])
        if expected is None:
            continue
        formatted_prompt = build_formatted_prompt(processor, prompt, None, reasoning_mode)
        inputs = process_inputs(processor, image, formatted_prompt, max_length=max_length, padding=False)
        inputs = move_batch_to_device(inputs, model_device(model))
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        raw_text = processor.tokenizer.decode(
            generated_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        correct += int(parse_answer(raw_text, "A") == expected)

    model.train()
    return {"accuracy": correct / total if total else 0.0, "count": float(total)}


def save_adapter(model: torch.nn.Module, processor: Any, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    save_processor(processor, checkpoint_dir)


def save_processor(processor: Any, output_dir: Path) -> None:
    if not hasattr(processor, "chat_template"):
        processor.chat_template = None
    try:
        processor.save_pretrained(output_dir)
    except AttributeError as exc:
        print(f"Warning: could not save Phi processor metadata: {exc}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_siglip2_filter_decorator()
    prompt = load_prompt(args.prompt_file)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if not hasattr(processor, "chat_template"):
        processor.chat_template = None
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"Loading train split: {args.dataset} [{args.train_split}]")
    train_dataset = load_dataset(args.dataset, split=args.train_split)
    eval_dataset = load_dataset(args.dataset, split=args.eval_split)

    id_column = pick_column(train_dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    image_column = pick_column(train_dataset.column_names, args.image_column, ["image", "image_id"])
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")

    train_dataset = maybe_limit(filter_dataset(train_dataset, args), args.train_limit)
    eval_dataset = maybe_limit(filter_dataset(eval_dataset, args), args.eval_limit)
    print(f"Train rows: {len(train_dataset)}")
    print(f"Eval rows: {len(eval_dataset)}")

    model = load_model(args)
    collator = VisualMcqCollator(
        processor=processor,
        prompt=prompt,
        image_column=image_column,
        answer_column=args.answer_column,
        max_length=args.max_length,
        image_variant=args.image_variant,
        enhance_longest_side=args.enhance_longest_side,
        reasoning_mode=args.reasoning_mode,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = int(math.ceil(args.num_epochs * updates_per_epoch))
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    training_args = vars(args)
    training_args.update(
        {
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        }
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(training_args, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=total_steps, desc="Training")
    while global_step < total_steps:
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, model_device(model))
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * args.grad_accum_steps
            micro_step += 1

            is_last_batch = batch_index == len(train_loader) - 1
            should_step = micro_step % args.grad_accum_steps == 0 or is_last_batch
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    running_loss = 0.0
                    progress.write(f"step={global_step} loss={avg_loss:.4f}")

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_adapter(model, processor, output_dir, global_step)

                if args.eval_steps > 0 and global_step % args.eval_steps == 0 and len(eval_dataset):
                    metrics = evaluate_generation(
                        model=model,
                        processor=processor,
                        dataset=eval_dataset,
                        prompt=prompt,
                        image_column=image_column,
                        answer_column=args.answer_column,
                        limit=args.eval_generate_limit,
                        max_new_tokens=args.max_new_tokens,
                        image_variant=args.image_variant,
                        enhance_longest_side=args.enhance_longest_side,
                        reasoning_mode=args.reasoning_mode,
                        max_length=args.max_length,
                    )
                    progress.write(
                        f"step={global_step} eval_accuracy={metrics['accuracy']:.4f} "
                        f"n={int(metrics['count'])}"
                    )

                if global_step >= total_steps:
                    break
        else:
            continue
        break

    progress.close()
    save_adapter(model, processor, output_dir, global_step)
    model.save_pretrained(output_dir)
    save_processor(processor, output_dir)
    print(f"Saved final LoRA adapter: {output_dir}")


if __name__ == "__main__":
    main()
