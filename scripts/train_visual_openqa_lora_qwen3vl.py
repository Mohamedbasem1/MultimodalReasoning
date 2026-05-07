import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image, ImageEnhance, ImageFilter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup

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
Do not output explanation, reasoning, chain-of-thought, or <think> tags."""
TOKENIZED_CHAT_PROCESSOR_KWARGS = {"padding": True, "return_tensors": "pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tune Qwen3-VL-8B-Thinking on Visual OpenQA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--init-adapter", default=None, help="Optional existing PEFT/LoRA adapter to continue training from.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--output-dir", default="outputs/qwen3vl8b-thinking-openqa-lora")
    parser.add_argument("--prompt-file", default="prompts/visual_openqa_prompt.txt")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="auto")
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--include-language", nargs="+", default=None)
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
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-answer-chars", type=int, default=1000)
    parser.add_argument("--enable-thinking", action="store_true", help="Allow Qwen3 thinking mode if supported.")
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-generate-limit", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1600)
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
    answer = re.sub(r"<think>.*?</think>", " ", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"</?think>", " ", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.strip(" \t\r\n\"'")
    if not answer or answer.lower() in {"none", "nan", "null", "hidden"}:
        return None
    if max_chars > 0 and len(answer) > max_chars:
        answer = answer[:max_chars].rstrip()
    return answer


def normalize_for_match(text: Any) -> str:
    value = str(text).lower()
    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.DOTALL)
    value = re.sub(r"[^\w\s.%/-]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


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


def build_messages(image: Image.Image, prompt: str, answer: Optional[str]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def apply_chat_template(
    processor: AutoProcessor,
    messages: List[List[Dict[str, Any]]] | List[Dict[str, Any]],
    enable_thinking: bool,
    **kwargs: Any,
) -> Any:
    try:
        return processor.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


class VisualOpenQaCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        prompt: str,
        image_column: str,
        answer_column: str,
        image_variant: str,
        enhance_longest_side: int,
        max_answer_chars: int,
        enable_thinking: bool,
    ):
        self.processor = processor
        self.prompt = prompt
        self.image_column = image_column
        self.answer_column = answer_column
        self.image_variant = image_variant
        self.enhance_longest_side = enhance_longest_side
        self.max_answer_chars = max_answer_chars
        self.enable_thinking = enable_thinking

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_messages: List[List[Dict[str, Any]]] = []
        prompt_messages: List[List[Dict[str, Any]]] = []

        for row in rows:
            image = normalize_image(row[self.image_column])
            image = select_image_variant(image, self.image_variant, self.enhance_longest_side)
            answer = clean_answer(row[self.answer_column], self.max_answer_chars)
            if answer is None:
                raise ValueError(f"Invalid answer: {row.get(self.answer_column)!r}")
            full_messages.append(build_messages(image, self.prompt, answer))
            prompt_messages.append(build_messages(image, self.prompt, None))

        full_inputs = apply_chat_template(
            processor=self.processor,
            messages=full_messages,
            enable_thinking=self.enable_thinking,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            processor_kwargs=TOKENIZED_CHAT_PROCESSOR_KWARGS,
        )
        prompt_inputs = apply_chat_template(
            processor=self.processor,
            messages=prompt_messages,
            enable_thinking=self.enable_thinking,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs=TOKENIZED_CHAT_PROCESSOR_KWARGS,
        )

        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[index, :prompt_length] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[full_inputs["input_ids"] == pad_token_id] = -100
        full_inputs["labels"] = labels
        return dict(full_inputs)


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
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = qwen3_from_pretrained(args.model, kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

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


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def model_device(model: torch.nn.Module) -> torch.device:
    model_device_attr = getattr(model, "device", None)
    if isinstance(model_device_attr, torch.device):
        return model_device_attr
    return next(model.parameters()).device


@torch.no_grad()
def evaluate_generation(
    model: torch.nn.Module,
    processor: AutoProcessor,
    dataset: Dataset,
    prompt: str,
    image_column: str,
    answer_column: str,
    limit: int,
    max_new_tokens: int,
    image_variant: str,
    enhance_longest_side: int,
    max_answer_chars: int,
    enable_thinking: bool,
) -> Dict[str, float]:
    model.eval()
    total = min(limit, len(dataset))
    exact = 0

    for row in tqdm(dataset.select(range(total)), desc="Eval generation", leave=False):
        image = normalize_image(row[image_column])
        image = select_image_variant(image, image_variant, enhance_longest_side)
        expected = clean_answer(row[answer_column], max_answer_chars)
        if expected is None:
            continue
        messages = build_messages(image, prompt, None)
        inputs = apply_chat_template(
            processor=processor,
            messages=messages,
            enable_thinking=enable_thinking,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={"return_tensors": "pt"},
        )
        inputs = move_batch_to_device(dict(inputs), model_device(model))
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        raw_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        prediction = clean_answer(raw_text, max_answer_chars) or ""
        exact += int(normalize_for_match(prediction) == normalize_for_match(expected))

    model.train()
    return {"exact_match": exact / total if total else 0.0, "count": float(total)}


def save_adapter(model: torch.nn.Module, processor: AutoProcessor, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(args.prompt_file)
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"Loading train split: {args.dataset} [{args.train_split}]")
    train_dataset = load_dataset(args.dataset, split=args.train_split)
    eval_dataset = load_dataset(args.dataset, split=args.eval_split)

    id_column = pick_column(train_dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    image_column = pick_column(train_dataset.column_names, args.image_column, ["image", "image_id"])
    answer_column = pick_column(
        train_dataset.column_names,
        args.answer_column,
        ["answer", "answer_text", "reference_answer", "gold_answer", "open_answer", "label"],
    )
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")
    print(f"Answer column: {answer_column}")

    train_dataset = maybe_limit(filter_dataset(train_dataset, args, answer_column), args.train_limit)
    eval_dataset = maybe_limit(filter_dataset(eval_dataset, args, answer_column), args.eval_limit)
    print(f"Train rows: {len(train_dataset)}")
    print(f"Eval rows: {len(eval_dataset)}")

    model = load_model(args)
    collator = VisualOpenQaCollator(
        processor=processor,
        prompt=prompt,
        image_column=image_column,
        answer_column=answer_column,
        image_variant=args.image_variant,
        enhance_longest_side=args.enhance_longest_side,
        max_answer_chars=args.max_answer_chars,
        enable_thinking=args.enable_thinking,
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
            "id_column": id_column,
            "image_column": image_column,
            "answer_column": answer_column,
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

                if args.eval_steps > 0 and global_step % args.eval_steps == 0 and len(eval_dataset):
                    metrics = evaluate_generation(
                        model=model,
                        processor=processor,
                        dataset=eval_dataset,
                        prompt=prompt,
                        image_column=image_column,
                        answer_column=answer_column,
                        limit=args.eval_generate_limit,
                        max_new_tokens=args.max_new_tokens,
                        image_variant=args.image_variant,
                        enhance_longest_side=args.enhance_longest_side,
                        max_answer_chars=args.max_answer_chars,
                        enable_thinking=args.enable_thinking,
                    )
                    progress.write(
                        f"step={global_step} eval_exact_match={metrics['exact_match']:.4f} "
                        f"n={int(metrics['count'])}"
                    )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_adapter(model, processor, output_dir, global_step)

                if global_step >= total_steps:
                    break
        else:
            continue
        break

    progress.close()
    save_adapter(model, processor, output_dir, global_step)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved final LoRA adapter: {output_dir}")


if __name__ == "__main__":
    main()
