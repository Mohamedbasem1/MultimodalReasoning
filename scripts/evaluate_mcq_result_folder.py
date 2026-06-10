#!/usr/bin/env python3
"""Evaluate all Visual MCQ prediction files in a folder.

This script is intentionally compatible with both the official MCQ format
(`id`, `answer_key`) and this repository's common format
(`question_id`, `answer_key`). It scans a directory such as `Result/`,
skips non-MCQ artifacts, and writes a JSON and CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ANSWER_KEYS = {"A", "B", "C", "D", "E"}
DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual"
DEFAULT_RESULT_DIR = "Result"
LANGUAGE_ISO2 = {
    "bulgarian": "bg",
    "chinese": "zh",
    "croatian": "hr",
    "english": "en",
    "italian": "it",
    "serbian": "sr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every MCQ prediction file in a Result folder.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--gold-file", default=None, help="Optional local gold JSON/JSONL file.")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--gold-column", default="auto")
    parser.add_argument("--language-column", default="auto")
    parser.add_argument("--page-size", type=int, default=100, help="Dataset Viewer page size for HTTP fallback.")
    parser.add_argument(
        "--glob",
        action="append",
        default=["*.json", "*.jsonl"],
        help="Glob(s) to scan. Can be passed multiple times.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include .raw.jsonl files. They often contain valid MCQ rows.",
    )
    parser.add_argument(
        "--strict-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require prediction ids to exactly match the gold split.",
    )
    parser.add_argument("--json-output", default="Result/mcq_folder_eval_summary.json")
    parser.add_argument("--csv-output", default="Result/mcq_folder_eval_summary.csv")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normalise_language(lang: Any) -> str:
    if lang is None:
        return "unknown"
    stripped = str(lang).strip()
    if not stripped:
        return "unknown"
    if len(stripped) == 2:
        return stripped.lower()
    return LANGUAGE_ISO2.get(stripped.lower(), stripped)


def is_correct_answer(pred_answer: str, gold_answer: str) -> bool:
    """Match the leaderboard behavior for multi-answer MCQ gold labels.

    Some released gold labels contain multiple acceptable letters, mostly in
    the Chinese subset (for example "BC" or "AD"). Submitted predictions are
    single letters, so count the prediction as correct when it is one of the
    acceptable gold letters.
    """

    pred_answer = str(pred_answer).strip().upper()
    gold_answer = str(gold_answer).strip().upper()
    if pred_answer == gold_answer:
        return True
    if pred_answer in ANSWER_KEYS and len(gold_answer) > 1:
        return pred_answer in set(gold_answer)
    return False


def pick_column(columns: Iterable[str], requested: str, candidates: Iterable[str], required: bool = True) -> str:
    columns = list(columns)
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"Column '{requested}' not found. Available: {columns}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"Could not infer column from candidates {list(candidates)}. Available: {columns}")
    return ""


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                rows.append(row)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("JSON root is not a list or {'data': [...]}")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError("not all JSON list items are objects")
    return data


def row_id(row: Dict[str, Any]) -> str | None:
    for key in ("question_id", "id", "sample_id"):
        if key in row:
            return str(row[key])
    return None


def row_answer(row: Dict[str, Any], first_char: bool = True) -> str | None:
    for key in ("answer_key", "answer", "prediction"):
        if key in row:
            answer = str(row[key]).strip().upper()
            if answer:
                return answer[:1] if first_char else answer
    return None


def load_predictions(path: Path) -> Dict[str, str]:
    rows = read_json_or_jsonl(path)
    if not rows:
        raise ValueError("empty file")

    predictions: Dict[str, str] = {}
    for index, row in enumerate(rows):
        qid = row_id(row)
        answer = row_answer(row, first_char=True)
        if qid is None or answer is None:
            raise ValueError(f"row {index} is missing id/question_id or answer_key/answer/prediction")
        if answer not in ANSWER_KEYS:
            raise ValueError(f"row {index} has unsupported answer {answer!r}")
        if qid in predictions:
            raise ValueError(f"duplicate id {qid!r}")
        predictions[qid] = answer
    return predictions


def load_gold_from_file(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    path = Path(args.gold_file)
    rows = read_json_or_jsonl(path)
    gold: Dict[str, Dict[str, str]] = {}
    for index, row in enumerate(rows):
        qid = row_id(row)
        answer = row_answer(row, first_char=False)
        if qid is None or answer is None:
            raise ValueError(f"Gold row {index} is missing id/question_id or answer_key/answer")
        gold[qid] = {
            "answer": answer,
            "language": normalise_language(row.get("language", "unknown")),
        }
    return gold, {"gold_file": str(path), "num_gold": len(gold)}


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()
    return None


def fetch_dataset_rows_via_viewer(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    token = hf_token()
    config = args.config or "default"
    rows: List[Dict[str, Any]] = []
    features: List[Dict[str, Any]] = []
    offset = 0
    page_size = max(1, min(int(args.page_size), 100))

    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": args.dataset,
                "config": config,
                "split": args.split,
                "offset": offset,
                "length": page_size,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if not features:
            features = payload.get("features", [])
        page_rows = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page_rows)
        if len(page_rows) < page_size:
            break
        offset += len(page_rows)

    metadata = {
        "dataset": args.dataset,
        "config": config,
        "split": args.split,
        "source": "datasets-server",
        "features": [feature.get("name") for feature in features],
        "num_rows": len(rows),
    }
    return rows, metadata


def build_gold_from_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    metadata: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    if not rows:
        raise ValueError("gold dataset returned no rows")
    columns = list(rows[0].keys())
    id_column = pick_column(columns, args.id_column, ["question_id", "id", "sample_id"])
    gold_column = pick_column(columns, args.gold_column, ["answer_key", "answer", "label"])
    language_column = pick_column(columns, args.language_column, ["language", "lang"], required=False)

    gold: Dict[str, Dict[str, str]] = {}
    for row in rows:
        qid = str(row[id_column])
        answer = str(row[gold_column]).strip().upper()
        gold[qid] = {
            "answer": answer,
            "language": normalise_language(row[language_column]) if language_column else "unknown",
        }

    metadata.update(
        {
            "num_gold": len(gold),
            "id_column": id_column,
            "gold_column": gold_column,
            "language_column": language_column or None,
        }
    )
    return gold, metadata


def load_gold_from_dataset(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        rows, metadata = fetch_dataset_rows_via_viewer(args)
        return build_gold_from_rows(rows, args, metadata)

    dataset_kwargs: Dict[str, Any] = {"path": args.dataset, "split": args.split}
    if args.config:
        dataset_kwargs["name"] = args.config
    dataset = load_dataset(**dataset_kwargs)

    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "id", "sample_id"])
    gold_column = pick_column(dataset.column_names, args.gold_column, ["answer_key", "answer", "label"])
    language_column = pick_column(
        dataset.column_names,
        args.language_column,
        ["language", "lang"],
        required=False,
    )

    metadata = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "source": "datasets",
        "id_column": id_column,
        "gold_column": gold_column,
        "language_column": language_column or None,
    }
    rows = [dict(row) for row in dataset]
    return build_gold_from_rows(rows, args, metadata)


def load_gold(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    if args.gold_file:
        return load_gold_from_file(args)
    return load_gold_from_dataset(args)


def evaluate_predictions(
    predictions: Dict[str, str],
    gold: Dict[str, Dict[str, str]],
    strict_size: bool,
) -> Dict[str, Any]:
    pred_ids = set(predictions)
    gold_ids = set(gold)
    missing = sorted(gold_ids - pred_ids)
    extra = sorted(pred_ids - gold_ids)
    duplicate_count = len(predictions) - len(pred_ids)

    if strict_size and (missing or extra):
        raise ValueError(
            f"id mismatch: missing={len(missing)}, extra={len(extra)}; "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )

    eval_ids = sorted(gold_ids & pred_ids)
    if not eval_ids:
        raise ValueError("no overlapping ids with gold split")

    correct = 0
    per_language: Dict[str, Dict[str, int]] = {}
    answer_distribution = {key: 0 for key in sorted(ANSWER_KEYS)}
    for qid in eval_ids:
        pred_answer = predictions[qid]
        gold_item = gold[qid]
        lang = gold_item["language"]
        answer_distribution[pred_answer] += 1
        per_language.setdefault(lang, {"correct": 0, "total": 0})
        per_language[lang]["total"] += 1
        if is_correct_answer(pred_answer, gold_item["answer"]):
            correct += 1
            per_language[lang]["correct"] += 1

    per_language_report = {
        lang: {
            "accuracy": round(counts["correct"] / counts["total"], 4) if counts["total"] else 0.0,
            "correct": counts["correct"],
            "num_samples": counts["total"],
        }
        for lang, counts in sorted(per_language.items())
    }

    return {
        "status": "ok",
        "accuracy": round(correct / len(eval_ids), 4),
        "correct": correct,
        "num_samples": len(eval_ids),
        "num_predictions": len(predictions),
        "missing_ids": len(missing),
        "extra_ids": len(extra),
        "duplicate_ids": duplicate_count,
        "answer_distribution": answer_distribution,
        "per_language": per_language_report,
    }


def is_probably_mcq_filename(path: Path, include_raw: bool) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return False
    if not include_raw and ".raw." in name:
        return False
    if name.endswith(".stats.json") or "compare" in name or "eval_summary" in name:
        return False
    if "openqa" in name:
        return False
    return "mcq" in name or path.suffix.lower() == ".jsonl"


def iter_candidate_files(args: argparse.Namespace) -> List[Path]:
    root = Path(args.result_dir)
    files = set()
    for pattern in args.glob:
        iterator = root.rglob(pattern) if args.recursive else root.glob(pattern)
        for path in iterator:
            if path.is_file() and is_probably_mcq_filename(path, args.include_raw):
                files.add(path)
    return sorted(files, key=lambda value: str(value).lower())


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "file",
        "status",
        "accuracy",
        "correct",
        "num_samples",
        "num_predictions",
        "missing_ids",
        "extra_ids",
        "error",
        "acc_bg",
        "acc_zh",
        "acc_hr",
        "acc_en",
        "acc_it",
        "acc_sr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = {
        "rank": row.get("rank", ""),
        "file": row["file"],
        "status": row["status"],
        "accuracy": row.get("accuracy", ""),
        "correct": row.get("correct", ""),
        "num_samples": row.get("num_samples", ""),
        "num_predictions": row.get("num_predictions", ""),
        "missing_ids": row.get("missing_ids", ""),
        "extra_ids": row.get("extra_ids", ""),
        "error": row.get("error", ""),
    }
    per_language = row.get("per_language") or {}
    for lang in ("bg", "zh", "hr", "en", "it", "sr"):
        flat[f"acc_{lang}"] = (per_language.get(lang) or {}).get("accuracy", "")
    return flat


def main() -> None:
    args = parse_args()
    gold, metadata = load_gold(args)
    candidate_files = iter_candidate_files(args)

    results: List[Dict[str, Any]] = []
    for path in candidate_files:
        result: Dict[str, Any] = {"file": str(path)}
        try:
            predictions = load_predictions(path)
            result.update(evaluate_predictions(predictions, gold, args.strict_size))
        except Exception as exc:  # keep scanning the folder
            result.update({"status": "skipped", "error": str(exc)})
        results.append(result)
        if args.verbose:
            if result["status"] == "ok":
                print(f"{path}: accuracy={result['accuracy']:.4f} ({result['correct']}/{result['num_samples']})")
            else:
                print(f"{path}: skipped: {result['error']}")

    ok_results = [row for row in results if row["status"] == "ok"]
    ok_results.sort(key=lambda row: (-row["accuracy"], row["file"]))
    for rank, row in enumerate(ok_results, start=1):
        row["rank"] = rank

    skipped_results = [row for row in results if row["status"] != "ok"]
    summary = {
        "gold": metadata,
        "result_dir": args.result_dir,
        "num_files_scanned": len(candidate_files),
        "num_evaluated": len(ok_results),
        "num_skipped": len(skipped_results),
        "results": ok_results + skipped_results,
    }

    json_output = Path(args.json_output)
    csv_output = Path(args.csv_output)
    atomic_write_json(json_output, summary)
    csv_rows = [flatten_for_csv(row) for row in ok_results + skipped_results]
    write_csv(csv_output, csv_rows)

    print(f"Scanned {len(candidate_files)} file(s); evaluated {len(ok_results)}; skipped {len(skipped_results)}.")
    if ok_results:
        best = ok_results[0]
        print(f"Best: {best['file']} accuracy={best['accuracy']:.4f} ({best['correct']}/{best['num_samples']})")
    print(f"Wrote JSON: {json_output}")
    print(f"Wrote CSV: {csv_output}")


if __name__ == "__main__":
    main()
