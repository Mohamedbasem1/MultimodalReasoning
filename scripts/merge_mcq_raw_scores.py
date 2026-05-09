import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


ANSWER_KEYS = ("A", "B", "C", "D", "E")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge MCQ raw JSONL files by per-answer score dictionaries."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Raw JSONL files from MCQ inference. Each row must contain question_id and scores.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-input weights. Example: --weights 1 2 biases toward the second file.",
    )
    parser.add_argument(
        "--method",
        choices=["weighted-sum", "confidence-router"],
        default="weighted-sum",
        help=(
            "weighted-sum adds weighted per-answer scores across files. "
            "confidence-router chooses the answer from the file with the largest top-vs-runner-up margin."
        ),
    )
    parser.add_argument("--output", required=True, help="Submission JSON output path.")
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional stats JSON path. Defaults to '<output>.stats.json'.",
    )
    return parser.parse_args()


def load_raw(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id", "")).strip()
            if not question_id:
                raise ValueError(f"{path}:{line_number} has no question_id")
            if question_id in rows:
                raise ValueError(f"{path} has duplicate question_id: {question_id}")
            scores = row.get("scores")
            if not isinstance(scores, dict):
                raise ValueError(f"{path}:{line_number} has no scores dict")
            parsed_scores = {}
            for answer in ANSWER_KEYS:
                if answer in scores:
                    parsed_scores[answer] = float(scores[answer])
            if not parsed_scores:
                raise ValueError(f"{path}:{line_number} has no usable A-E scores")
            answer_key = str(row.get("answer_key", "")).strip().upper()
            if answer_key not in ANSWER_KEYS:
                answer_key = max(parsed_scores, key=parsed_scores.get)
            rows[question_id] = {
                "answer_key": answer_key,
                "scores": parsed_scores,
            }
    return rows


def top_answer(scores: Dict[str, float]) -> Tuple[str, float]:
    answer = max(scores, key=scores.get)
    return answer, scores[answer]


def confidence_margin(scores: Dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return ordered[0] - ordered[1]


def weighted_sum_choice(rows: Sequence[Dict[str, Any]], weights: Sequence[float]) -> Tuple[str, Dict[str, float], List[str]]:
    combined = {answer: 0.0 for answer in ANSWER_KEYS}
    votes = []
    for row, weight in zip(rows, weights):
        votes.append(str(row["answer_key"]))
        scores = row["scores"]
        for answer in ANSWER_KEYS:
            if answer in scores:
                combined[answer] += weight * float(scores[answer])
            else:
                combined[answer] += weight * -1e9
    return max(combined, key=combined.get), combined, votes


def confidence_router_choice(rows: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, float], List[str], int]:
    margins = [confidence_margin(row["scores"]) for row in rows]
    best_index = max(range(len(rows)), key=lambda index: margins[index])
    return rows[best_index]["answer_key"], rows[best_index]["scores"], [row["answer_key"] for row in rows], best_index


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    weights = args.weights if args.weights is not None else [1.0] * len(input_paths)
    if len(weights) != len(input_paths):
        raise ValueError(f"--weights length ({len(weights)}) must match --inputs length ({len(input_paths)})")

    raw_maps = [load_raw(path) for path in input_paths]
    reference_ids = list(raw_maps[0])
    reference_set = set(reference_ids)
    for path, raw_map in zip(input_paths[1:], raw_maps[1:]):
        ids = set(raw_map)
        missing = sorted(reference_set - ids)
        extra = sorted(ids - reference_set)
        if missing or extra:
            raise ValueError(
                f"{path} IDs do not match first input. missing={len(missing)} extra={len(extra)} "
                f"missing_examples={missing[:5]} extra_examples={extra[:5]}"
            )

    merged_rows: List[Dict[str, str]] = []
    detail_rows: List[Dict[str, Any]] = []
    changed_from_first = 0
    changed_from_last = 0
    agreement_count = 0
    chosen_source_counts: Counter[str] = Counter()

    for question_id in reference_ids:
        rows = [raw_map[question_id] for raw_map in raw_maps]
        if args.method == "weighted-sum":
            answer_key, merged_scores, votes = weighted_sum_choice(rows, weights)
            chosen_source = "score_ensemble"
        else:
            answer_key, merged_scores, votes, source_index = confidence_router_choice(rows)
            chosen_source = str(input_paths[source_index])
            chosen_source_counts[chosen_source] += 1

        merged_rows.append({"question_id": question_id, "answer_key": answer_key})
        changed_from_first += int(answer_key != rows[0]["answer_key"])
        changed_from_last += int(answer_key != rows[-1]["answer_key"])
        agreement_count += int(len(set(votes)) == 1)
        top_score = merged_scores.get(answer_key)
        detail_rows.append(
            {
                "question_id": question_id,
                "answer_key": answer_key,
                "input_answers": votes,
                "chosen_source": chosen_source,
                "top_score": top_score,
                "scores": merged_scores,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "method": args.method,
        "inputs": [str(path) for path in input_paths],
        "weights": weights,
        "rows": len(merged_rows),
        "agreement_count": agreement_count,
        "disagreement_count": len(merged_rows) - agreement_count,
        "answer_distribution": dict(sorted(Counter(row["answer_key"] for row in merged_rows).items())),
        "input_distributions": [
            dict(sorted(Counter(row["answer_key"] for row in raw_map.values()).items()))
            for raw_map in raw_maps
        ],
        "changed_from_first": changed_from_first,
        "changed_from_last": changed_from_last,
        "chosen_source_counts": dict(sorted(chosen_source_counts.items())),
        "examples": detail_rows[:30],
    }
    stats_output_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote merged submission: {output_path}")
    print(f"Wrote stats: {stats_output_path}")
    print(f"Rows: {len(merged_rows)}")
    print(f"Method: {args.method}")
    print(f"Weights: {weights}")
    print(f"Input agreement: {agreement_count}/{len(merged_rows)}")
    print(f"Answer distribution: {stats['answer_distribution']}")
    print(f"Changed from first input: {changed_from_first}")
    print(f"Changed from last input: {changed_from_last}")
    if chosen_source_counts:
        print(f"Chosen source counts: {dict(sorted(chosen_source_counts.items()))}")


if __name__ == "__main__":
    main()
