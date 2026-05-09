import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_ID_COLUMNS = ("question_id", "sample_id", "id")
DEFAULT_ANSWER_COLUMNS = ("answer_key", "answer", "prediction", "predicted_answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two MCQ prediction JSON files.")
    parser.add_argument("file_a", help="First prediction JSON file.")
    parser.add_argument("file_b", help="Second prediction JSON file.")
    parser.add_argument("--name-a", default=None, help="Display name for the first file.")
    parser.add_argument("--name-b", default=None, help="Display name for the second file.")
    parser.add_argument("--id-column", default="auto", help="ID column name, or auto.")
    parser.add_argument("--answer-column", default="auto", help="Answer column name, or auto.")
    parser.add_argument("--output", default=None, help="Optional JSON output with summary and disagreements.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output with disagreement rows.")
    parser.add_argument("--show", type=int, default=30, help="Number of disagreements to print.")
    return parser.parse_args()


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path} row {index} must be an object.")
        rows.append(row)
    return rows


def pick_column(row: Dict[str, Any], requested: str, candidates: Sequence[str], label: str) -> str:
    if requested != "auto":
        if requested not in row:
            raise ValueError(f"Requested {label} column '{requested}' not found. Available: {sorted(row)}")
        return requested
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise ValueError(f"Could not infer {label} column. Available: {sorted(row)}")


def normalize_answer(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    stripped = text.strip("()[]{} .:-")
    if stripped in {"A", "B", "C", "D", "E"}:
        return stripped
    return text


def rows_to_map(rows: List[Dict[str, Any]], id_column: str, answer_column: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    duplicates = []
    for row in rows:
        question_id = str(row[id_column])
        if question_id in result:
            duplicates.append(question_id)
        result[question_id] = normalize_answer(row[answer_column])
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"Duplicate IDs found: {preview}")
    return result


def print_distribution(name: str, values: Iterable[str]) -> None:
    counts = Counter(values)
    total = sum(counts.values())
    print(f"{name}:")
    for answer, count in sorted(counts.items()):
        pct = 100 * count / total if total else 0
        print(f"  {answer}: {count} ({pct:.2f}%)")


def main() -> None:
    args = parse_args()
    path_a = Path(args.file_a)
    path_b = Path(args.file_b)
    name_a = args.name_a or path_a.stem
    name_b = args.name_b or path_b.stem

    rows_a = load_json_rows(path_a)
    rows_b = load_json_rows(path_b)
    if not rows_a:
        raise ValueError(f"{path_a} has no rows.")
    if not rows_b:
        raise ValueError(f"{path_b} has no rows.")

    id_col_a = pick_column(rows_a[0], args.id_column, DEFAULT_ID_COLUMNS, "ID")
    id_col_b = pick_column(rows_b[0], args.id_column, DEFAULT_ID_COLUMNS, "ID")
    answer_col_a = pick_column(rows_a[0], args.answer_column, DEFAULT_ANSWER_COLUMNS, "answer")
    answer_col_b = pick_column(rows_b[0], args.answer_column, DEFAULT_ANSWER_COLUMNS, "answer")

    map_a = rows_to_map(rows_a, id_col_a, answer_col_a)
    map_b = rows_to_map(rows_b, id_col_b, answer_col_b)
    ids_a = set(map_a)
    ids_b = set(map_b)
    common_ids = sorted(ids_a & ids_b)
    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)

    agreements = [question_id for question_id in common_ids if map_a[question_id] == map_b[question_id]]
    disagreements = [question_id for question_id in common_ids if map_a[question_id] != map_b[question_id]]
    pair_counts = Counter((map_a[question_id], map_b[question_id]) for question_id in common_ids)

    print("Files")
    print(f"  {name_a}: {path_a} rows={len(rows_a)} id={id_col_a} answer={answer_col_a}")
    print(f"  {name_b}: {path_b} rows={len(rows_b)} id={id_col_b} answer={answer_col_b}")
    print()
    print("ID coverage")
    print(f"  common: {len(common_ids)}")
    print(f"  only {name_a}: {len(only_a)}")
    print(f"  only {name_b}: {len(only_b)}")
    print()
    print("Distributions")
    print_distribution(name_a, map_a.values())
    print_distribution(name_b, map_b.values())
    print()
    print("Agreement")
    agreement_rate = len(agreements) / len(common_ids) if common_ids else 0
    print(f"  agree: {len(agreements)}")
    print(f"  disagree: {len(disagreements)}")
    print(f"  agreement_rate: {agreement_rate:.4f}")
    print()
    print(f"Pair counts ({name_a} -> {name_b})")
    for (answer_a, answer_b), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {answer_a} -> {answer_b}: {count}")

    if args.show:
        print()
        print(f"First {min(args.show, len(disagreements))} disagreements")
        for question_id in disagreements[: args.show]:
            print(f"  {question_id}: {map_a[question_id]} vs {map_b[question_id]}")

    disagreement_rows = [
        {
            "question_id": question_id,
            name_a: map_a[question_id],
            name_b: map_b[question_id],
        }
        for question_id in disagreements
    ]
    summary = {
        "file_a": str(path_a),
        "file_b": str(path_b),
        "name_a": name_a,
        "name_b": name_b,
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "common": len(common_ids),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "agree": len(agreements),
        "disagree": len(disagreements),
        "agreement_rate": agreement_rate,
        "distribution_a": dict(sorted(Counter(map_a.values()).items())),
        "distribution_b": dict(sorted(Counter(map_b.values()).items())),
        "pair_counts": [
            {"a": answer_a, "b": answer_b, "count": count}
            for (answer_a, answer_b), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "only_a_ids": only_a,
        "only_b_ids": only_b,
        "disagreements": disagreement_rows,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON: {output_path}")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question_id", name_a, name_b])
            writer.writeheader()
            writer.writerows(disagreement_rows)
        print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
