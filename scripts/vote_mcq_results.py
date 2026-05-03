import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ANSWER_KEYS = {"A", "B", "C", "D", "E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Majority-vote multiple ImageCLEF Visual MCQ submission JSON files."
    )
    parser.add_argument(
        "--input-dir",
        default="Result",
        help="Folder containing prediction JSON files. Used when --inputs is omitted.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Prediction JSON files in priority order. First file wins all-different ties.",
    )
    parser.add_argument(
        "--output",
        default="Result/imageclef_visual_mcq_voted.json",
        help="Output voted submission JSON path.",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional stats JSON path. Defaults to '<output>.stats.json'.",
    )
    return parser.parse_args()


def discover_inputs(input_dir: str, output_path: Path) -> List[Path]:
    folder = Path(input_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    output_resolved = output_path.resolve()
    paths = []
    for path in sorted(folder.glob("*.json")):
        if path.resolve() == output_resolved:
            continue
        if path.name.endswith(".stats.json"):
            continue
        paths.append(path)

    if len(paths) < 2:
        raise ValueError(f"Need at least 2 JSON files to vote. Found {len(paths)} in {folder}.")
    return paths


def load_prediction_file(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    rows = []
    seen = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} row {index} is not an object.")
        question_id = str(item.get("question_id", "")).strip()
        answer_key = str(item.get("answer_key", "")).strip().upper()
        if not question_id:
            raise ValueError(f"{path} row {index} has empty question_id.")
        if answer_key not in ANSWER_KEYS:
            raise ValueError(f"{path} row {index} has invalid answer_key: {answer_key!r}.")
        if question_id in seen:
            raise ValueError(f"{path} has duplicate question_id: {question_id}")
        seen.add(question_id)
        rows.append({"question_id": question_id, "answer_key": answer_key})
    return rows


def load_all(paths: Sequence[Path]) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, str]]]:
    file_names = [str(path) for path in paths]
    all_rows = [load_prediction_file(path) for path in paths]
    reference_ids = [row["question_id"] for row in all_rows[0]]

    for path, rows in zip(paths[1:], all_rows[1:]):
        question_ids = [row["question_id"] for row in rows]
        if question_ids != reference_ids:
            missing = sorted(set(reference_ids) - set(question_ids))[:5]
            extra = sorted(set(question_ids) - set(reference_ids))[:5]
            raise ValueError(
                f"{path} does not have the same question_id order as {paths[0]}. "
                f"missing_examples={missing} extra_examples={extra}"
            )

    return file_names, all_rows[0], [dict((row["question_id"], row["answer_key"]) for row in rows) for rows in all_rows]


def vote_answer(question_id: str, lookups: Sequence[Dict[str, str]]) -> Tuple[str, str, Dict[str, int]]:
    answers = [lookup[question_id] for lookup in lookups]
    counts = Counter(answers)
    top_count = max(counts.values())
    top_answers = {answer for answer, count in counts.items() if count == top_count}

    if len(top_answers) == 1:
        answer = next(iter(top_answers))
        mode = "unanimous" if top_count == len(answers) else "majority"
        return answer, mode, dict(sorted(counts.items()))

    # With 3 files this means all three disagree. Keep the first input as priority.
    return answers[0], "tie_first_input", dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")

    input_paths = [Path(path) for path in args.inputs] if args.inputs else discover_inputs(args.input_dir, output_path)
    file_names, reference_rows, lookups = load_all(input_paths)

    voted = []
    mode_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    disagreements = []

    for row in reference_rows:
        question_id = row["question_id"]
        answer_key, mode, counts = vote_answer(question_id, lookups)
        voted.append({"question_id": question_id, "answer_key": answer_key})
        mode_counts[mode] += 1
        answer_counts[answer_key] += 1
        if mode != "unanimous":
            disagreements.append(
                {
                    "question_id": question_id,
                    "chosen_answer": answer_key,
                    "decision_mode": mode,
                    "votes": counts,
                }
            )

    stats = {
        "inputs": file_names,
        "output": str(output_path),
        "rows": len(voted),
        "decision_modes": dict(sorted(mode_counts.items())),
        "answer_distribution": dict(sorted(answer_counts.items())),
        "num_disagreements": len(disagreements),
        "disagreement_examples": disagreements[:20],
    }

    output_path.write_text(json.dumps(voted, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Inputs: {len(input_paths)}")
    for path in input_paths:
        print(f"- {path}")
    print(f"Rows: {len(voted)}")
    print(f"Decision modes: {dict(sorted(mode_counts.items()))}")
    print(f"Answer distribution: {dict(sorted(answer_counts.items()))}")
    print(f"Wrote voted submission: {output_path}")
    print(f"Wrote stats: {stats_output_path}")


if __name__ == "__main__":
    main()
