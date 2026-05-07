import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Visual OpenQA predictions to official submission format.")
    parser.add_argument("predictions", help="Input JSON with question_id + answer, or already answer-like fields.")
    parser.add_argument("--output", required=True, help="Output official JSON path.")
    parser.add_argument("--dataset", default="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--page-size", type=int, default=100)
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list.")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Input row {index} must be an object.")
        rows.append(row)
    return rows


def fetch_dataset_rows(dataset: str, config: str, split: str, page_size: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    total = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        with urllib.request.urlopen(f"{DATASET_SERVER_ROWS_URL}?{query}") as response:
            payload = json.loads(response.read().decode("utf-8"))
        total = int(payload["num_rows_total"])
        for item in payload["rows"]:
            rows.append(item["row"])
        offset += page_size
    return rows


def extract_answer(row: Dict[str, Any]) -> str:
    if "answers" in row:
        answers = row["answers"]
        if isinstance(answers, list):
            return str(answers[0]).strip() if answers else ""
        return str(answers).strip()
    if "answer" in row:
        return str(row["answer"]).strip()
    return ""


def build_answer_map(rows: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    answer_map: Dict[str, str] = {}
    for index, row in enumerate(rows):
        question_id = str(row.get("question_id", row.get("id", ""))).strip()
        if not question_id:
            raise ValueError(f"Prediction row {index} has no question_id/id.")
        if question_id in answer_map:
            raise ValueError(f"Duplicate prediction id: {question_id}")
        answer_map[question_id] = extract_answer(row)
    return answer_map


def main() -> None:
    args = parse_args()
    predictions = load_json(Path(args.predictions))
    answer_map = build_answer_map(predictions)
    dataset_rows = fetch_dataset_rows(args.dataset, args.config, args.split, args.page_size)

    submission: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in dataset_rows:
        question_id = str(row["question_id"])
        seen_ids.add(question_id)
        answer = answer_map.get(question_id, "")
        submission.append(
            {
                "question_id": question_id,
                "answers": [answer],
                "language": str(row.get("language", "")),
            }
        )

    extra_ids = set(answer_map) - seen_ids
    if extra_ids:
        raise ValueError(f"Predictions contain {len(extra_ids)} IDs that are not in {args.split}.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote official OpenQA submission: {output_path}")
    print(f"Rows: {len(submission)}")


if __name__ == "__main__":
    main()
