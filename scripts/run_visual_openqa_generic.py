import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor

from run_visual_mcq_gemma4 import (
    build_generate_inputs,
    decode_response,
    load_model,
    load_prompt,
    model_device,
    model_float_dtype,
    move_batch_to_device,
    normalize_image,
    select_image_variant,
)


DEFAULT_MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generic vision-language model on Visual OpenQA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/visual_openqa_generic.json")
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
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None, choices=[None, "sdpa", "eager", "flash_attention_2"])
    parser.add_argument(
        "--model-loader",
        choices=["auto", "image-text-to-text", "causal-lm"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--image-variant", default="enhanced", choices=["original", "enhanced"])
    parser.add_argument("--enhance-longest-side", type=int, default=1000)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


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


def export_gold(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote gold file: {path}")


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


def clean_answer(raw_text: str, max_chars: int) -> str:
    candidates = [raw_text.strip()]
    if re.search(r"</think>", raw_text, flags=re.IGNORECASE):
        candidates.extend(part.strip() for part in re.split(r"</think>", raw_text, flags=re.IGNORECASE) if part.strip())

    cleaned_candidates = []
    for text in candidates:
        if re.search(r"final\s+answer\s*:", text, flags=re.IGNORECASE):
            text = re.split(r"final\s+answer\s*:", text, flags=re.IGNORECASE)[-1]
        text = re.sub(r"<think>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<answer>|</answer>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:final\s+answer|answer)\s*(?:is|:|-)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:the\s+)?user\s+wants\s+me\s+to\s+[^.:\n]*(?:\.|:)\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:image\s+analysis|problem\s+analysis|analysis)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip(" \t\r\n\"'")
        if text:
            cleaned_candidates.append(text)
    text = max(cleaned_candidates, key=len) if cleaned_candidates else ""
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def normalize_for_match(text: Any) -> str:
    value = str(text).lower()
    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.DOTALL)
    value = re.sub(r"[^\w\s.%/-]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Inference will be very slow on CPU.")

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
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args)
    device = model_device(model)
    input_float_dtype = model_float_dtype(model)

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
        for row in tqdm(dataset, desc="Generic-OpenQA"):
            question_id = str(row[id_column])
            if question_id in completed_ids:
                continue

            language = str(row.get("language", "")) if has_language else ""
            image = select_image_variant(normalize_image(row[image_column]), args.image_variant, args.enhance_longest_side)
            inputs = move_batch_to_device(
                build_generate_inputs(processor, image, prompt),
                device,
                input_float_dtype,
            )
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )

            raw_text = decode_response(processor, generated_ids[0], input_len)
            answer = clean_answer(raw_text, args.max_answer_chars)
            if not answer:
                answer = "N/A"

            predictions.append(
                {
                    "question_id": question_id,
                    "answers": [answer],
                    "language": language,
                }
            )
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
