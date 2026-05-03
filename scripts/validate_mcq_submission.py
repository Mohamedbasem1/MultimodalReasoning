import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import load_dataset


ANSWER_KEYS = {"A", "B", "C", "D", "E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ImageCLEF Visual MCQ JSON submission.")
    parser.add_argument("submission")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--filter-type", nargs="+", default=None)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow predictions for only part of the dataset. Useful for smoke tests.",
    )
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
    submission_path = Path(args.submission)
    rows = load_submission(submission_path)

    seen_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object.")
        if set(row.keys()) != {"question_id", "answer_key"}:
            raise ValueError(f"Row {index} must contain exactly question_id and answer_key.")
        question_id = str(row["question_id"])
        answer_key = str(row["answer_key"]).strip().upper()
        if not question_id:
            raise ValueError(f"Row {index} has an empty question_id.")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        if answer_key not in ANSWER_KEYS:
            raise ValueError(f"Invalid answer_key for {question_id}: {row['answer_key']!r}")
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

    if args.answer_column in dataset.column_names:
        pred_by_id = {str(row["question_id"]): str(row["answer_key"]).strip().upper() for row in rows}
        correct = 0
        scored = 0
        for row in dataset:
            question_id = str(row[id_column])
            if question_id not in pred_by_id:
                continue
            gold = str(row[args.answer_column]).strip().upper()
            if gold in ANSWER_KEYS:
                scored += 1
                correct += int(pred_by_id[question_id] == gold)
        if scored:
            print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")


if __name__ == "__main__":
    main()

