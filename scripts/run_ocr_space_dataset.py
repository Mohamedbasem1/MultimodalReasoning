#!/usr/bin/env python3
"""Run OCR.space over a Hugging Face image dataset split.

The script is deliberately resumable: it appends JSONL rows as each sample is
processed and skips ids already present in the output file on restart.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Set

import requests
from datasets import load_dataset
from PIL import Image


LANGUAGE_TO_OCR_SPACE = {
    "bulgarian": "bul",
    "bg": "bul",
    "chinese": "chs",
    "zh": "chs",
    "croatian": "hrv",
    "hr": "hrv",
    "english": "eng",
    "en": "eng",
    "italian": "ita",
    "it": "ita",
    # OCR.space does not list Serbian in the public docs. Engine 2/3 auto is
    # usually the least bad choice for unsupported languages.
    "serbian": "auto",
    "sr": "auto",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR a Hugging Face image dataset split with OCR.space.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--language-column", default="auto")
    parser.add_argument("--output", required=True, help="JSONL path to append OCR results.")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--api-key", default=os.environ.get("OCR_SPACE_API_KEY"))
    parser.add_argument("--endpoint", default="https://api.ocr.space/parse/image")
    parser.add_argument("--engine", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--language", default="by-row", help="'by-row', 'auto', or OCR.space language code.")
    parser.add_argument("--is-table", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detect-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resize-longest-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-bytes", type=int, default=950_000, help="Keep uploads below this size when possible.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--index-file",
        default=None,
        help="Optional JSON/TXT file of dataset indexes to process instead of a contiguous start/limit range.",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Do not treat previous error rows as completed when resuming.",
    )
    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help="Do not treat previous rows with empty OCR text as completed when resuming.",
    )
    return parser.parse_args()


def pick_column(columns: Iterable[str], requested: str, candidates: Iterable[str]) -> str:
    columns = list(columns)
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"Column '{requested}' not found. Available: {columns}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not infer column from {list(candidates)}. Available: {columns}")


def completed_ids(path: Path, retry_errors: bool = False, retry_empty: bool = False) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_errors and (row.get("error") or row.get("is_errored_on_processing")):
                continue
            if retry_empty and not str(row.get("text") or row.get("ocr_text") or "").strip():
                continue
            if "id" in row:
                ids.add(str(row["id"]))
    return ids


def load_index_file(path: str) -> list[int]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("indexes", "indices", "missing_or_failed_indexes"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"Index file JSON must be a list or contain an indexes list: {path}")
        return [int(item) for item in data]
    indexes: list[int] = []
    for part in text.replace(",", "\n").splitlines():
        part = part.strip()
        if part:
            indexes.append(int(part))
    return indexes


def normalize_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, dict) and image.get("bytes") is not None:
        img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, (str, os.PathLike)):
        img = Image.open(image)
    else:
        raise TypeError(f"Unsupported image value: {type(image)!r}")
    return img.convert("RGB")


def encode_image(image: Image.Image, longest_side: int, jpeg_quality: int, max_bytes: int) -> bytes:
    img = image.copy()
    if longest_side and max(img.size) > longest_side:
        img.thumbnail((longest_side, longest_side), Image.Resampling.LANCZOS)

    quality = jpeg_quality
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        payload = buffer.getvalue()
        if len(payload) <= max_bytes or quality <= 45:
            return payload
        quality -= 10


def resolve_language(row: Dict[str, Any], language_column: str, requested: str) -> str:
    if requested != "by-row":
        return requested
    language = str(row.get(language_column, "auto")).strip().lower()
    return LANGUAGE_TO_OCR_SPACE.get(language, "auto")


def parsed_text(response_json: Dict[str, Any]) -> str:
    parts = []
    for result in response_json.get("ParsedResults") or []:
        text = result.get("ParsedText")
        if text:
            parts.append(str(text).strip())
    return "\n\n".join(part for part in parts if part)


def call_ocr_space(
    *,
    endpoint: str,
    api_key: str,
    image_bytes: bytes,
    filename: str,
    language: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    data = {
        "language": language,
        "isOverlayRequired": str(args.overlay).lower(),
        "OCREngine": str(args.engine),
        "scale": str(args.scale).lower(),
        "detectOrientation": str(args.detect_orientation).lower(),
        "isTable": str(args.is_table).lower(),
    }
    headers = {"apikey": api_key}
    files = {"file": (filename, image_bytes, "image/jpeg")}

    last_exc: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, data=data, files=files, timeout=args.timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # requests can raise multiple concrete types.
            last_exc = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)
                continue
    raise RuntimeError(f"OCR.space request failed after {args.max_retries} attempts: {last_exc}") from last_exc


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing OCR.space API key. Set OCR_SPACE_API_KEY or pass --api-key.")

    dataset_kwargs: Dict[str, Any] = {"path": args.dataset, "split": args.split}
    if args.config:
        dataset_kwargs["name"] = args.config
    dataset = load_dataset(**dataset_kwargs)

    columns = dataset.column_names
    id_column = pick_column(columns, args.id_column, ["question_id", "id", "sample_id"])
    image_column = pick_column(columns, args.image_column, ["image"])
    language_column = pick_column(columns, args.language_column, ["language"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output, retry_errors=args.retry_errors, retry_empty=args.retry_empty)

    total_rows = len(dataset)
    if args.index_file:
        indexes = [idx for idx in load_index_file(args.index_file) if 0 <= idx < total_rows]
        if args.limit is not None:
            indexes = indexes[: args.limit]
        stop = None
    else:
        stop = total_rows if args.limit is None else min(total_rows, args.start + args.limit)
        indexes = list(range(args.start, stop))
    written = 0
    skipped = 0
    errors = 0

    with output.open("a", encoding="utf-8") as handle:
        for index in indexes:
            row = dataset[index]
            item_id = str(row[id_column])
            if item_id in done:
                skipped += 1
                continue

            language = resolve_language(row, language_column, args.language)
            try:
                image = normalize_image(row[image_column])
                image_bytes = encode_image(image, args.resize_longest_side, args.jpeg_quality, args.max_bytes)
                response_json = call_ocr_space(
                    endpoint=args.endpoint,
                    api_key=args.api_key,
                    image_bytes=image_bytes,
                    filename=f"{item_id}.jpg",
                    language=language,
                    args=args,
                )
                result = {
                    "id": item_id,
                    "index": index,
                    "language": row.get(language_column),
                    "ocr_language": language,
                    "subject": row.get("subject"),
                    "text": parsed_text(response_json),
                    "is_errored_on_processing": response_json.get("IsErroredOnProcessing"),
                    "ocr_exit_code": response_json.get("OCRExitCode"),
                    "error_message": response_json.get("ErrorMessage"),
                    "processing_time_ms": response_json.get("ProcessingTimeInMilliseconds"),
                    "raw": response_json,
                }
            except Exception as exc:
                errors += 1
                result = {
                    "id": item_id,
                    "index": index,
                    "language": row.get(language_column),
                    "subject": row.get("subject"),
                    "text": "",
                    "error": str(exc),
                }

            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(item_id)
            written += 1
            print(f"[{index + 1}/{total_rows}] {item_id} lang={result.get('ocr_language')} chars={len(result.get('text', ''))}")
            if args.sleep:
                time.sleep(args.sleep)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "id_column": id_column,
        "image_column": image_column,
        "language_column": language_column,
        "output": str(output),
        "rows_in_dataset": total_rows,
        "start": args.start,
        "stop": stop,
        "index_file": args.index_file,
        "num_requested_indexes": len(indexes),
        "new_rows_written": written,
        "already_completed_skipped": skipped,
        "errors": errors,
        "total_completed_in_output": len(done),
    }
    summary_output = Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
