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
from transformers import AutoModelForCausalLM, AutoProcessor

from run_visual_mcq_gemma4 import (
    ANSWER_KEYS,
    decode_response,
    dtype_from_arg,
    load_prompt,
    normalize_for_match,
    normalize_image,
    option_token_ids,
    parse_answer,
    pick_column,
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


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs = model_kwargs(args)
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
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args)
    add_image_preprocess = getattr(model, "add_image_preprocess", None)
    if callable(add_image_preprocess):
        add_image_preprocess(processor)
        print("Registered ERNIE image preprocessing.")
    repaired = repair_meta_rotary_tensors(model)
    if repaired:
        print(f"Repaired {repaired} meta rotary tensor(s).")

    device = model_device(model)
    token_ids_by_answer = option_token_ids(processor)
    if args.selection_method == "logits":
        print(f"Using next-token MCQ scoring with option token IDs: {token_ids_by_answer}")

    predictions: List[Dict[str, str]] = []
    correct = 0
    scored = 0

    with raw_output_path.open("w", encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="ERNIE4.5-MCQ"):
            question_id = str(row[id_column])
            image = select_image_variant(normalize_image(row[image_column]), args.image_variant, args.enhance_longest_side)
            temp_image_path = save_temp_image(image)

            try:
                with torch.inference_mode():
                    if args.selection_method == "logits":
                        inputs = move_inputs(build_inputs(processor, temp_image_path, prompt, args.answer_prefill), device)
                        answer_key, scores = score_answer_logits(model, inputs, token_ids_by_answer, args.fallback_answer)
                        raw_text = ""
                    else:
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
                        answer_key = parse_answer(raw_text, args.fallback_answer)
                        scores = {}
            finally:
                if os.path.exists(temp_image_path):
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
