import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from run_visual_mcq_gemma4 import (
    build_generate_inputs,
    decode_response,
    dtype_from_arg,
    enable_meta_nonzero_fallback,
    model_device,
    normalize_image,
    patch_ernie_vision_forward,
    repair_ernie_moe_meta_masks,
    repair_meta_rotary_tensors,
    save_temp_image_for_processor,
    select_image_variant,
)
from run_visual_openqa_generic import clean_answer, normalize_for_match


DEFAULT_MODEL = "baidu/ERNIE-4.5-VL-28B-A3B-Thinking"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERNIE-4.5-VL on ImageCLEF Visual OpenQA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/openqa_ernie45_vl_28b_thinking.json")
    parser.add_argument("--raw-output", default=None, help="Defaults to '<output>.raw.jsonl'.")
    parser.add_argument("--gold-output", default=None, help="Optional gold JSON path for labeled splits.")
    parser.add_argument("--prompt-file", default="prompts/visual_openqa_prompt.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-answer-chars", type=int, default=1000)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=768)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def load_prompt(path: str) -> str:
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "You are answering a visual open-ended exam question.\n\n"
        "Read the image carefully, including all question text, diagrams, charts, "
        "tables, labels, formulas, and units.\n\n"
        "Output only the concise final answer text in the same language as the question when possible."
    )


def pick_column(columns: Sequence[str], requested: str, candidates: Iterable[str], required: bool = True) -> str:
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"Column '{requested}' not found. Available: {list(columns)}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"Could not infer column. Available: {list(columns)}")
    return ""


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


def move_inputs(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def write_json_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [row for row in data if isinstance(row, dict)]


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: ignoring malformed raw JSONL line {line_number} in {path}")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_existing_predictions(output_path: Path, raw_output_path: Path) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in read_json_rows(output_path):
        question_id = str(row.get("question_id", "")).strip()
        if not question_id or question_id in seen_ids:
            continue
        answers = row.get("answers")
        if not isinstance(answers, list):
            answer = str(row.get("answer", "")).strip() or "N/A"
            answers = [answer]
        predictions.append(
            {
                "question_id": question_id,
                "answers": [str(answer) for answer in answers],
                "language": str(row.get("language", "")),
            }
        )
        seen_ids.add(question_id)

    for row in read_jsonl_rows(raw_output_path):
        question_id = str(row.get("question_id", "")).strip()
        if not question_id or question_id in seen_ids:
            continue
        answers = row.get("answers")
        if not isinstance(answers, list):
            answer = str(row.get("answer", "")).strip() or "N/A"
            answers = [answer]
        predictions.append(
            {
                "question_id": question_id,
                "answers": [str(answer) for answer in answers],
                "language": str(row.get("language", "")),
            }
        )
        seen_ids.add(question_id)
    return predictions


def export_gold(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote gold file: {path}")


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
    answer_column = pick_column(
        dataset.column_names,
        args.answer_column,
        ["answer", "answer_text", "reference_answer", "gold_answer", "open_answer", "label"],
        required=False,
    )
    has_language = "language" in dataset.column_names

    print(f"Rows: {len(dataset)}")
    print(f"ID column: {id_column}")
    print(f"Image column: {image_column}")
    if answer_column:
        print(f"Gold column: {answer_column}")

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

    if args.resume:
        predictions = load_existing_predictions(output_path, raw_output_path)
        completed_ids = {str(row["question_id"]) for row in predictions}
        if predictions:
            print(f"Resume enabled: found {len(predictions)} completed prediction(s).")
            write_json_atomic(output_path, predictions)
    else:
        predictions = []
        completed_ids = set()
        if output_path.exists():
            output_path.unlink()
        if raw_output_path.exists():
            raw_output_path.unlink()

    gold_rows: List[Dict[str, Any]] = []
    exact = 0
    scored = 0
    raw_mode = "a" if args.resume else "w"

    with raw_output_path.open(raw_mode, encoding="utf-8") as raw_file:
        for row in tqdm(dataset, desc="ERNIE4.5-OpenQA"):
            question_id = str(row[id_column])
            if question_id in completed_ids:
                continue

            language = str(row.get("language", "")) if has_language else ""
            image = select_image_variant(
                normalize_image(row[image_column]),
                args.image_variant,
                args.enhance_longest_side,
            )
            temp_image_path = save_temp_image_for_processor(image, processor)
            try:
                inputs = move_inputs(
                    build_generate_inputs(processor, image, prompt, image_path=temp_image_path),
                    device,
                )
                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=False,
                    )
            finally:
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

            raw_text = decode_response(processor, generated_ids[0], input_len)
            answer = clean_answer(raw_text, args.max_answer_chars)
            if not answer:
                answer = "N/A"

            prediction = {
                "question_id": question_id,
                "answers": [answer],
                "language": language,
            }
            predictions.append(prediction)
            completed_ids.add(question_id)

            raw_row: Dict[str, Any] = {
                "question_id": question_id,
                "answers": [answer],
                "answer": answer,
                "language": language,
                "raw_text": raw_text,
            }
            if answer_column:
                gold = str(row[answer_column]).strip()
                raw_row["gold"] = gold
                gold_rows.append(
                    {
                        "question_id": question_id,
                        "answers": [gold],
                        "language": language,
                    }
                )
                if gold and gold.upper() != "HIDDEN":
                    scored += 1
                    exact += int(normalize_for_match(answer) == normalize_for_match(gold))

            raw_file.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
            raw_file.flush()
            write_json_atomic(output_path, predictions)

    write_json_atomic(output_path, predictions)
    print(f"Wrote predictions: {output_path}")
    print(f"Wrote raw outputs: {raw_output_path}")
    if args.gold_output and gold_rows:
        export_gold(gold_rows, Path(args.gold_output))
    if scored:
        print(f"Exact match: {exact / scored:.4f} ({exact}/{scored})")


if __name__ == "__main__":
    main()
