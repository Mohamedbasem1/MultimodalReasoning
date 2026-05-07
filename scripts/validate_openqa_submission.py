import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ImageCLEF Visual OpenQA JSON submission.")
    parser.add_argument("submission")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--allow-subset", action="store_true")
    return parser.parse_args()


def pick_column(columns: Sequence[str], requested: str, candidates: Iterable[str]) -> str:
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"Column '{requested}' not found. Available: {list(columns)}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not infer id column. Available: {list(columns)}")


def load_submission(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Submission must be a JSON list.")
    return data


def main() -> None:
    args = parse_args()
    rows = load_submission(Path(args.submission))

    seen_ids = set()
    required_keys = {"question_id", args.answer_field}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object.")
        if set(row.keys()) != required_keys:
            raise ValueError(f"Row {index} must contain exactly question_id and {args.answer_field}.")
        question_id = str(row["question_id"])
        answer = str(row[args.answer_field]).strip()
        if not question_id:
            raise ValueError(f"Row {index} has an empty question_id.")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        if not answer:
            raise ValueError(f"Empty answer for {question_id}.")
        seen_ids.add(question_id)

    print(f"Basic JSON validation passed: {len(rows)} predictions")
    if not args.dataset:
        return

    dataset = load_dataset(args.dataset, split=args.split)
    if args.filter_type:
        allowed_types = set(args.filter_type)
        dataset = dataset.filter(lambda row: row.get("type") in allowed_types)

    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    gold_ids = {str(row[id_column]) for row in dataset}
    missing = gold_ids - seen_ids
    extra = seen_ids - gold_ids

    if missing and not args.allow_subset:
        raise ValueError(f"Submission is missing {len(missing)} dataset IDs.")
    if extra:
        raise ValueError(f"Submission has {len(extra)} IDs not present in the dataset.")
    if not args.allow_subset and len(rows) != len(dataset):
        raise ValueError(f"Submission size {len(rows)} does not match dataset size {len(dataset)}.")

    print(f"Dataset validation passed against {args.dataset} [{args.split}]")


if __name__ == "__main__":
    main()
