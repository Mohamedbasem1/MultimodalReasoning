import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from datasets import load_dataset

from run_visual_mcq_qwen25 import ANSWER_KEYS, pick_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route between two Visual MCQ prediction files using dataset metadata."
    )
    parser.add_argument("--primary", required=True, help="Default prediction JSON.")
    parser.add_argument("--secondary", required=True, help="Prediction JSON used for routed weak cases.")
    parser.add_argument("--dataset", default="MBZUAI/EXAMS-V")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument(
        "--route-languages",
        nargs="+",
        default=["Arabic", "Urdu"],
        help="Use secondary predictions for these language values when the dataset has a language column.",
    )
    parser.add_argument(
        "--route-types",
        nargs="+",
        default=["image_text"],
        help="Use secondary predictions for these type values when the dataset has a type column.",
    )
    parser.add_argument(
        "--route-binary-columns",
        nargs="+",
        default=["graph", "table"],
        help="Use secondary predictions when these metadata columns are 1/true.",
    )
    parser.add_argument(
        "--route-subject-contains",
        nargs="+",
        default=[],
        help="Use secondary predictions when subject contains any of these case-insensitive substrings.",
    )
    return parser.parse_args()


def load_predictions(path: str) -> Dict[str, str]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    predictions = {}
    for row in rows:
        question_id = str(row["question_id"])
        answer_key = str(row["answer_key"]).strip().upper()
        if answer_key not in ANSWER_KEYS:
            raise ValueError(f"Invalid answer_key in {path} for {question_id}: {answer_key}")
        predictions[question_id] = answer_key
    return predictions


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def should_route(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if "language" in row and str(row["language"]) in set(args.route_languages):
        return True
    if "type" in row and str(row["type"]) in set(args.route_types):
        return True
    for column in args.route_binary_columns:
        if column in row and truthy(row[column]):
            return True
    if args.route_subject_contains and "subject" in row:
        subject = str(row["subject"]).lower()
        if any(fragment.lower() in subject for fragment in args.route_subject_contains):
            return True
    return False


def main() -> None:
    args = parse_args()
    primary = load_predictions(args.primary)
    secondary = load_predictions(args.secondary)
    dataset = load_dataset(args.dataset, split=args.split)
    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    has_gold = args.answer_column in dataset.column_names

    output_rows: List[Dict[str, str]] = []
    routed = 0
    correct = 0
    scored = 0
    route_correct = 0
    route_total = 0
    primary_correct = 0
    primary_total = 0

    for row in dataset:
        question_id = str(row[id_column])
        if question_id not in primary:
            continue
        use_secondary = should_route(row, args) and question_id in secondary
        answer_key = secondary[question_id] if use_secondary else primary[question_id]
        routed += int(use_secondary)
        output_rows.append({"question_id": question_id, "answer_key": answer_key})

        if has_gold:
            gold = str(row[args.answer_column]).strip().upper()
            if gold in ANSWER_KEYS:
                scored += 1
                is_correct = answer_key == gold
                correct += int(is_correct)
                if use_secondary:
                    route_total += 1
                    route_correct += int(is_correct)
                else:
                    primary_total += 1
                    primary_correct += int(is_correct)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote routed predictions: {output_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Routed to secondary: {routed}")
    if scored:
        print(f"Accuracy: {correct / scored:.4f} ({correct}/{scored})")
        if route_total:
            print(f"Secondary-routed accuracy: {route_correct / route_total:.4f} ({route_correct}/{route_total})")
        if primary_total:
            print(f"Primary-kept accuracy: {primary_correct / primary_total:.4f} ({primary_correct}/{primary_total})")


if __name__ == "__main__":
    main()

