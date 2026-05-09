import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from run_visual_mcq_gemma4 import (
    ANSWER_KEYS,
    decode_response,
    dtype_from_arg,
    enable_meta_nonzero_fallback,
    load_prompt,
    normalize_for_match,
    normalize_image,
    option_token_ids,
    patch_ernie_vision_forward,
    parse_answer,
    pick_column,
    repair_ernie_moe_meta_masks,
    repair_meta_rotary_tensors,
    score_answer_logits,
    select_image_variant,
)


DEFAULT_MODEL = "baidu/ERNIE-4.5-VL-28B-A3B-Thinking"
DEFAULT_DATASET = "MBZUAI/EXAMS-V"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERNIE-4.5-VL on Visual MCQ datasets.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/ernie45_vl_mcq.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--prompt-file", default="prompts/visual_mcq_final_only.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--fallback-answer", choices=sorted(ANSWER_KEYS), default="A")
    parser.add_argument("--selection-method", choices=["logits", "generate"], default="logits")
    parser.add_argument("--answer-prefill", default="The correct option is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for logits scoring. Start with 2 for ERNIE.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=768)
    return parser.parse_args()


def model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
        "dtype": dtype_from_arg(args.torch_dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def load_config(args: argparse.Namespace) -> Any:
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if args.model.lower().endswith("-pt") and getattr(config, "multimodel_experts", False):
        if not getattr(config, "moe_use_hard_gate", False):
            config.moe_use_hard_gate = True
            print("Patched ERNIE PT config: moe_use_hard_gate=True")
    return config


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs = model_kwargs(args)
    kwargs["config"] = load_config(args)
    try:
        return AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    except TypeError as exc:
        if "dtype" not in str(exc) or "dtype" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(args.model, **retry_kwargs).eval()


def model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return parameter.device
    for buffer in model.buffers():
        if buffer.device.type == "cuda":
            return buffer.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_temp_image(image: Image.Image) -> str:
    handle = tempfile.NamedTemporaryFile(prefix="ernie45_vl_", suffix=".png", delete=False)
    temp_path = handle.name
    handle.close()
    image.save(temp_path)
    return temp_path


def build_messages(image_path: str, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_path}},
            ],
        }
    ]


def apply_ernie_chat_template(processor: Any, messages: List[Dict[str, Any]]) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_inputs(processor: Any, image_path: str, prompt: str, answer_prefill: str) -> Dict[str, Any]:
    messages = build_messages(image_path, prompt)
    text = apply_ernie_chat_template(processor, messages) + answer_prefill
    image_inputs, video_inputs = processor.process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def extend_optional_list(target: List[Any], values: Optional[Any]) -> None:
    if values is None:
        return
    if isinstance(values, list):
        target.extend(values)
    else:
        target.append(values)


def build_batch_inputs(processor: Any, image_paths: Sequence[str], prompt: str, answer_prefill: str) -> Dict[str, Any]:
    texts = []
    all_image_inputs: List[Any] = []
    all_video_inputs: List[Any] = []
    for image_path in image_paths:
        messages = build_messages(image_path, prompt)
        texts.append(apply_ernie_chat_template(processor, messages) + answer_prefill)
        image_inputs, video_inputs = processor.process_vision_info(messages)
        extend_optional_list(all_image_inputs, image_inputs)
        extend_optional_list(all_video_inputs, video_inputs)

    return processor(
        text=texts,
        images=all_image_inputs or None,
        videos=all_video_inputs or None,
        padding=True,
        return_tensors="pt",
    )


def build_generate_inputs(processor: Any, image_path: str, prompt: str) -> Dict[str, Any]:
    messages = build_messages(image_path, prompt)
    text = apply_ernie_chat_template(processor, messages)
    image_inputs, video_inputs = processor.process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def move_inputs(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def align_token_type_ids_for_forward(inputs: Dict[str, Any]) -> Dict[str, Any]:
    input_ids = inputs.get("input_ids")
    token_type_ids = inputs.get("token_type_ids")
    if not torch.is_tensor(input_ids) or not torch.is_tensor(token_type_ids):
        return inputs
    if token_type_ids.shape[-1] != input_ids.shape[-1]:
        return inputs
    extra = torch.zeros(
        (*token_type_ids.shape[:-1], 1),
        dtype=token_type_ids.dtype,
        device=token_type_ids.device,
    )
    inputs["token_type_ids"] = torch.cat([token_type_ids, extra], dim=-1)
    return inputs


def score_answer_logits_batch(
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    token_ids_by_answer: Dict[str, List[int]],
    fallback: str,
) -> Tuple[List[str], List[Dict[str, float]]]:
    outputs = model(**inputs, use_cache=False, return_dict=True)
    logits = outputs.logits.float()
    attention_mask = inputs.get("attention_mask")
    if torch.is_tensor(attention_mask):
        positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
        last_positions = (attention_mask.to(logits.device).long() * positions).max(dim=1).values
    else:
        last_positions = torch.full((logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device)

    row_logits = logits[torch.arange(logits.shape[0], device=logits.device), last_positions]
    log_probs = torch.log_softmax(row_logits, dim=-1)
    answers: List[str] = []
    score_rows: List[Dict[str, float]] = []
    for row_index in range(log_probs.shape[0]):
        scores: Dict[str, float] = {}
        for answer, token_ids in token_ids_by_answer.items():
            valid_ids = [token_id for token_id in token_ids if token_id < log_probs.shape[-1]]
            if valid_ids:
                scores[answer] = float(log_probs[row_index, valid_ids].max().item())
        answers.append(max(scores, key=scores.get) if scores else fallback)
        score_rows.append(scores)
    return answers, score_rows


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
    if enable_meta_nonzero_fallback():
        print("Enabled PyTorch meta nonzero fallback.")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args)
    add_image_preprocess = getattr(model, "add_image_preprocess", None)
    if callable(add_image_preprocess):
        add_image_preprocess(processor)
        print("Registered ERNIE image preprocessing.")
    if patch_ernie_vision_forward(model):
        print("Patched ERNIE vision preprocessing.")
    repaired = repair_meta_rotary_tensors(model)
    if repaired:
        print(f"Repaired {repaired} meta rotary tensor(s).")
    repaired_moe_masks = repair_ernie_moe_meta_masks(model)
    if repaired_moe_masks:
        print(f"Repaired {repaired_moe_masks} ERNIE MoE expert mask module(s).")

    device = model_device(model)
    token_ids_by_answer = option_token_ids(processor)
    if args.selection_method == "logits":
        print(f"Using next-token MCQ scoring with option token IDs: {token_ids_by_answer}")

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    batch_size = max(1, args.batch_size if args.selection_method == "logits" else 1)
    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        with tqdm(total=len(dataset), desc="ERNIE4.5-MCQ") as progress:
            for start in range(0, len(dataset), batch_size):
                rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
                question_ids = [str(row[id_column]) for row in rows]
                temp_image_paths = []
                try:
                    for row in rows:
                        image = select_image_variant(
                            normalize_image(row[image_column]),
                            args.image_variant,
                            args.enhance_longest_side,
                        )
                        temp_image_paths.append(save_temp_image(image))

                    with torch.inference_mode():
                        if args.selection_method == "logits":
                            inputs = align_token_type_ids_for_forward(
                                build_batch_inputs(processor, temp_image_paths, prompt, args.answer_prefill)
                            )
                            inputs = move_inputs(inputs, device)
                            answer_keys, score_rows = score_answer_logits_batch(
                                model,
                                inputs,
                                token_ids_by_answer,
                                args.fallback_answer,
                            )
                            raw_texts = [""] * len(rows)
                        else:
                            answer_keys = []
                            score_rows = []
                            raw_texts = []
                            for temp_image_path in temp_image_paths:
                                inputs = move_inputs(build_generate_inputs(processor, temp_image_path, prompt), device)
                                input_len = inputs["input_ids"].shape[-1]
                                outputs = model.generate(
                                    inputs=inputs["input_ids"],
                                    **inputs,
                                    do_sample=False,
                                    max_new_tokens=args.max_new_tokens,
                                    use_cache=False,
                                )
                                raw_text = decode_response(processor, outputs[0], input_len)
                                raw_texts.append(raw_text)
                                answer_keys.append(parse_answer(raw_text, args.fallback_answer))
                                score_rows.append({})
                finally:
                    for temp_image_path in temp_image_paths:
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)

                for row, question_id, answer_key, scores, raw_text in zip(
                    rows,
                    question_ids,
                    answer_keys,
                    score_rows,
                    raw_texts,
                ):
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
                raw_file.flush()
                progress.update(len(rows))

    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw outputs: {raw_output_path}")
    if scored:
        print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")


if __name__ == "__main__":
    main()
