import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from transformers.dynamic_module_utils import get_class_from_dynamic_module


ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_MODEL = "openbmb/MiniCPM-V-4_5"
DEFAULT_DATASET = "MBZUAI/EXAMS-V"
DEFAULT_PROMPT = """You are solving a visual multiple-choice exam question.

Read the image carefully, including all question text, answer options, diagrams, charts, tables, labels, formulas, and units.

Think internally if needed, but output only the final option letter: A, B, C, D, or E.
Do not output explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tune MiniCPM-V 4.5 on labeled Visual MCQ data.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision.")
    parser.add_argument("--init-adapter", default=None, help="Optional existing PEFT/LoRA adapter to continue training from.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--output-dir", default="outputs/minicpm-v45-examsv-lora")
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--filter-type", nargs="+", default=None, help="Optional values for the dataset 'type' column.")
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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-inp-length", type=int, default=4096)
    parser.add_argument("--max-slice-nums", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100, help="Set to 0 to disable generation eval during training.")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-generate-limit", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Recommended for single-GPU QLoRA.")
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Enable Transformers low-memory/meta loading. Leave off if MiniCPM loading hits compatibility errors.",
    )
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


def dtype_from_arg(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def patch_minicpm_tied_weight_attr(model_class: Any) -> None:
    if hasattr(model_class, "all_tied_weights_keys"):
        return

    def all_tied_weights_keys(self: torch.nn.Module) -> Dict[str, Any]:
        tied_keys = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied_keys, dict):
            return tied_keys
        if isinstance(tied_keys, (list, tuple, set)):
            return {str(key): None for key in tied_keys}
        return {}

    model_class.all_tied_weights_keys = property(all_tied_weights_keys)


def get_remote_model_class(model_name: str, revision: Optional[str]) -> Any:
    config_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if revision:
        config_kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)
    auto_map = getattr(config, "auto_map", {}) or {}
    class_reference = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if not class_reference:
        return None
    try:
        return get_class_from_dynamic_module(class_reference, model_name, revision=revision)
    except TypeError:
        return get_class_from_dynamic_module(class_reference, model_name)


def load_remote_model_with_patch(args: argparse.Namespace, kwargs: Dict[str, Any]) -> torch.nn.Module:
    model_class = get_remote_model_class(args.model, args.revision)
    if model_class is None:
        return AutoModel.from_pretrained(args.model, **kwargs)
    patch_minicpm_tied_weight_attr(model_class)
    return model_class.from_pretrained(args.model, **kwargs)


def ensure_minicpm_tokenizer_attrs(tokenizer: Any) -> Any:
    for alias_attr, token_id_attr in [("bos_id", "bos_token_id"), ("eos_id", "eos_token_id")]:
        if not hasattr(tokenizer, alias_attr) and getattr(tokenizer, token_id_attr, None) is not None:
            setattr(tokenizer, alias_attr, getattr(tokenizer, token_id_attr))
    return tokenizer


def token_id_for(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token) if hasattr(tokenizer, "convert_tokens_to_ids") else None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
        return token_id
    try:
        encoded = tokenizer.encode(token, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(token)
    if not encoded:
        raise ValueError(f"Could not encode MiniCPM special token: {token}")
    return int(encoded[0])


def patch_minicpm_processor(processor: Any) -> Any:
    tokenizer = ensure_minicpm_tokenizer_attrs(processor.tokenizer)
    image_processor = processor.image_processor
    token_attrs = {
        "im_start_id": getattr(image_processor, "im_start_token", "<image>"),
        "im_end_id": getattr(image_processor, "im_end_token", "</image>"),
        "slice_start_id": getattr(image_processor, "slice_start_token", "<slice>"),
        "slice_end_id": getattr(image_processor, "slice_end_token", "</slice>"),
    }
    for attr_name, token in token_attrs.items():
        if hasattr(tokenizer, attr_name):
            continue
        setattr(tokenizer, attr_name, token_id_for(tokenizer, token))
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer
    return processor


def load_processor(model_name: str, revision: Optional[str]) -> Any:
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    return patch_minicpm_processor(processor)


def apply_minicpm_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_minicpm_text_and_images(
    processor: Any,
    image: Image.Image,
    prompt: str,
    answer: Optional[str],
    add_generation_prompt: bool,
) -> Tuple[str, List[Image.Image]]:
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": ["(<image>./</image>)", prompt]},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    text = apply_minicpm_chat_template(processor.tokenizer, messages, add_generation_prompt)
    return text, [image]


def processor_call(
    processor: Any,
    texts: List[str],
    images: List[List[Image.Image]],
    max_inp_length: int,
    max_slice_nums: Optional[int],
) -> Dict[str, Any]:
    inputs = processor(
        texts,
        images,
        max_slice_nums=max_slice_nums,
        return_tensors="pt",
        max_length=max_inp_length,
    )
    return dict(inputs)


def add_position_ids(inputs: Dict[str, Any]) -> Dict[str, Any]:
    attention_mask = inputs["attention_mask"].long()
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    inputs["position_ids"] = position_ids
    return inputs


class VisualMcqCollator:
    def __init__(
        self,
        processor: Any,
        prompt: str,
        image_column: str,
        answer_column: str,
        max_inp_length: int,
        max_slice_nums: Optional[int],
    ):
        self.processor = processor
        self.prompt = prompt
        self.image_column = image_column
        self.answer_column = answer_column
        self.max_inp_length = max_inp_length
        self.max_slice_nums = max_slice_nums

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        full_images: List[List[Image.Image]] = []
        prompt_texts: List[str] = []
        prompt_images: List[List[Image.Image]] = []

        for row in rows:
            image = normalize_image(row[self.image_column])
            answer = normalize_answer(row[self.answer_column])
            if answer is None:
                raise ValueError(f"Invalid answer: {row.get(self.answer_column)!r}")
            full_text, full_image_list = build_minicpm_text_and_images(
                self.processor,
                image,
                self.prompt,
                answer,
                add_generation_prompt=False,
            )
            prompt_text, prompt_image_list = build_minicpm_text_and_images(
                self.processor,
                image,
                self.prompt,
                None,
                add_generation_prompt=True,
            )
            full_texts.append(full_text)
            full_images.append(full_image_list)
            prompt_texts.append(prompt_text)
            prompt_images.append(prompt_image_list)

        full_inputs = processor_call(
            self.processor,
            full_texts,
            full_images,
            self.max_inp_length,
            self.max_slice_nums,
        )
        prompt_inputs = processor_call(
            self.processor,
            prompt_texts,
            prompt_images,
            self.max_inp_length,
            self.max_slice_nums,
        )
        full_inputs = add_position_ids(full_inputs)

        labels = full_inputs["input_ids"].clone().long()
        full_lengths = full_inputs["attention_mask"].sum(dim=1)
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        sequence_length = labels.shape[1]
        for index, (full_length, prompt_length) in enumerate(zip(full_lengths.tolist(), prompt_lengths.tolist())):
            start = sequence_length - int(full_length)
            labels[index, start : start + int(prompt_length)] = -100

        labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels
        full_inputs.pop("image_sizes", None)
        return full_inputs


def parse_answer(raw_text: str, fallback: str = "A") -> str:
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


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": dtype_from_arg(args.torch_dtype),
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if args.load_in_4bit:
        kwargs["device_map"] = args.device_map
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = load_remote_model_with_patch(args, kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif torch.cuda.is_available():
        model = model.cuda()

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


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: move_to_device(value, device) for key, value in batch.items()}


def split_minicpm_forward_batch(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_data_keys = {"input_ids", "pixel_values", "image_bound", "tgt_sizes", "position_ids", "temporal_ids"}
    data = {key: batch.pop(key) for key in list(batch.keys()) if key in model_data_keys}
    batch.pop("image_sizes", None)
    return data, batch


def unwrap_for_chat(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


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
    max_slice_nums: Optional[int],
) -> Dict[str, float]:
    model.eval()
    chat_model = unwrap_for_chat(model)
    total = min(limit, len(dataset))
    correct = 0

    for row in tqdm(dataset.select(range(total)), desc="Eval generation", leave=False):
        image = normalize_image(row[image_column])
        expected = normalize_answer(row[answer_column])
        if expected is None:
            continue
        raw_text = chat_model.chat(
            msgs=[{"role": "user", "content": [image, prompt]}],
            tokenizer=processor.tokenizer,
            processor=processor,
            sampling=False,
            max_new_tokens=max_new_tokens,
            max_slice_nums=max_slice_nums,
            enable_thinking=False,
        )
        correct += int(parse_answer(str(raw_text)) == expected)

    model.train()
    return {"accuracy": correct / total if total else 0.0, "count": float(total)}


def save_adapter(model: torch.nn.Module, processor: Any, output_dir: Path, step: int) -> None:
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
    processor = load_processor(args.model, args.revision)

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
        max_inp_length=args.max_inp_length,
        max_slice_nums=args.max_slice_nums,
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
            data, kwargs = split_minicpm_forward_batch(batch)
            outputs = model(data=data, **kwargs)
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
                        max_slice_nums=args.max_slice_nums,
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
    processor.save_pretrained(output_dir)
    print(f"Saved final LoRA adapter: {output_dir}")


if __name__ == "__main__":
    main()
