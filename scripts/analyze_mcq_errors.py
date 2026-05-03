import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image

from run_visual_mcq_qwen25 import ANSWER_KEYS, normalize_image, pick_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Visual MCQ prediction errors.")
    parser.add_argument("predictions")
    parser.add_argument("--dataset", default="MBZUAI/EXAMS-V")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/error_analysis")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--answer-column", default="answer_key")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--max-unique", type=int, default=80)
    parser.add_argument("--min-group-size", type=int, default=10)
    parser.add_argument("--sample-errors", type=int, default=100)
    return parser.parse_args()


def load_predictions(path: str) -> Dict[str, str]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["question_id"]): str(row["answer_key"]).strip().upper() for row in rows}


def load_raw(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    records = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record.get("question_id", ""))] = record
    return records


def scalar_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text or None
    if isinstance(value, list) and len(value) <= 4 and all(isinstance(item, (str, int, float, bool)) for item in value):
        return "|".join(str(item).strip() for item in value)
    return None


def image_bucket(image: Image.Image) -> Dict[str, str]:
    width, height = image.size
    longest = max(width, height)
    aspect = width / height if height else 0.0
    if longest < 700:
        size_bucket = "small"
    elif longest < 1200:
        size_bucket = "medium"
    elif longest < 1800:
        size_bucket = "large"
    else:
        size_bucket = "xlarge"
    if aspect < 0.75:
        aspect_bucket = "portrait"
    elif aspect > 1.35:
        aspect_bucket = "landscape"
    else:
        aspect_bucket = "squareish"
    return {
        "image_size_bucket": size_bucket,
        "image_aspect_bucket": aspect_bucket,
        "image_longest_side_bucket": f"{(longest // 500) * 500}-{((longest // 500) + 1) * 500}",
    }


def summarize_groups(groups: Dict[str, Dict[str, List[int]]], min_group_size: int) -> Dict[str, List[Dict[str, Any]]]:
    summary = {}
    for column, values in groups.items():
        rows = []
        for value, (correct, total) in values.items():
            if total >= min_group_size:
                rows.append(
                    {
                        "value": value,
                        "accuracy": correct / total if total else 0.0,
                        "correct": correct,
                        "total": total,
                    }
                )
        rows.sort(key=lambda row: (row["accuracy"], -row["total"]))
        if rows:
            summary[column] = rows
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args.predictions)
    raw_records = load_raw(args.raw_output)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    id_column = pick_column(dataset.column_names, args.id_column, ["question_id", "sample_id", "id"])
    image_column = pick_column(dataset.column_names, args.image_column, ["image", "image_id"])

    scalar_columns = []
    sample = dataset.select(range(min(200, len(dataset))))
    for column in dataset.column_names:
        if column in {id_column, image_column, args.answer_column}:
            continue
        values = [scalar_value(row.get(column)) for row in sample]
        values = [value for value in values if value is not None]
        if values and len(set(values)) <= args.max_unique:
            scalar_columns.append(column)

    total = 0
    correct_count = 0
    confusion = {gold: {pred: 0 for pred in sorted(ANSWER_KEYS)} for gold in sorted(ANSWER_KEYS)}
    pred_distribution = Counter()
    gold_distribution = Counter()
    groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    errors = []

    for row in dataset:
        question_id = str(row[id_column])
        if question_id not in predictions:
            continue
        pred = predictions[question_id]
        gold = str(row[args.answer_column]).strip().upper()
        if gold not in ANSWER_KEYS:
            continue
        total += 1
        is_correct = pred == gold
        correct_count += int(is_correct)
        pred_distribution[pred] += 1
        gold_distribution[gold] += 1
        if pred in ANSWER_KEYS:
            confusion[gold][pred] += 1

        for column in scalar_columns:
            value = scalar_value(row.get(column))
            if value is not None:
                groups[column][value][0] += int(is_correct)
                groups[column][value][1] += 1
        try:
            for column, value in image_bucket(normalize_image(row[image_column])).items():
                groups[column][value][0] += int(is_correct)
                groups[column][value][1] += 1
        except Exception:
            pass

        if not is_correct and len(errors) < args.sample_errors:
            error_record = {
                "question_id": question_id,
                "pred": pred,
                "gold": gold,
                "metadata": {column: scalar_value(row.get(column)) for column in scalar_columns},
            }
            if question_id in raw_records:
                raw = raw_records[question_id]
                for key in ["raw_text", "vote_counts", "votes", "aggregate_scores"]:
                    if key in raw:
                        error_record[key] = raw[key]
            errors.append(error_record)

    summary = {
        "predictions": args.predictions,
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "pred_distribution": dict(sorted(pred_distribution.items())),
        "gold_distribution": dict(sorted(gold_distribution.items())),
        "confusion": confusion,
        "group_accuracy": summarize_groups(groups, args.min_group_size),
        "scalar_columns": scalar_columns,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    print(f"Accuracy: {summary['accuracy']:.4f} ({correct_count}/{total})")
    print(f"Wrote: {output_dir / 'summary.json'}")
    print(f"Wrote: {output_dir / 'errors.jsonl'}")
    for column, rows in list(summary["group_accuracy"].items())[:12]:
        print(f"\n{column}")
        for row in rows[:5]:
            print(f"  {row['value']}: {row['accuracy']:.4f} ({row['correct']}/{row['total']})")


if __name__ == "__main__":
    main()

