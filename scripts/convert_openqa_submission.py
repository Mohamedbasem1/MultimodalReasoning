import argparse
import json
import re
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
    parser.add_argument(
        "--split-answers",
        action="store_true",
        help="Split obvious multi-part answers into an answers list.",
    )
    parser.add_argument(
        "--keep-leading-labels",
        action="store_true",
        help="Keep leading option/subpart labels such as 'A)' or 'Б)'.",
    )
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


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_leading_label(text: str) -> str:
    stripped = normalize_space(text)
    stripped = re.sub(
        r"^(?:answer|final answer)\s*(?:is|:|-)?\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"^(?:[A-EА-Еа-е]|[IVX]{1,4}|[ivx]{1,4}|\d{1,2})\s*[\)\].:：-]\s*",
        "",
        stripped,
    )
    return stripped.strip(" \t\r\n\"'")


def split_answer_text(answer: str, keep_leading_labels: bool) -> List[str]:
    answer = normalize_space(answer)
    if not answer or answer.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return [""]

    pieces = [answer]
    marker_pattern = r"(?=(?:^|\s)(?:[A-EА-Еа-е]|\d{1,2})\s*[\)\].:：]\s+)"
    marker_splits = [part.strip() for part in re.split(marker_pattern, answer) if part.strip()]
    if len(marker_splits) > 1:
        pieces = marker_splits
    elif ";" in answer:
        semicolon_splits = [part.strip() for part in answer.split(";") if part.strip()]
        if 1 < len(semicolon_splits) <= 8:
            pieces = semicolon_splits

    cleaned: List[str] = []
    for piece in pieces:
        value = normalize_space(piece)
        if not keep_leading_labels:
            value = strip_leading_label(value)
        if value:
            cleaned.append(value)
    return cleaned or [""]


def extract_answers(row: Dict[str, Any], split_answers: bool, keep_leading_labels: bool) -> List[str]:
    if "answers" in row:
        answers = row["answers"]
        if isinstance(answers, list):
            raw_answers = [str(answer).strip() for answer in answers]
        else:
            raw_answers = [str(answers).strip()]
    elif "answer" in row:
        raw_answers = [str(row["answer"]).strip()]
    else:
        raw_answers = [""]

    output: List[str] = []
    for answer in raw_answers:
        if split_answers:
            output.extend(split_answer_text(answer, keep_leading_labels))
        else:
            value = normalize_space(answer)
            if not keep_leading_labels:
                value = strip_leading_label(value)
            if value.upper() in {"N/A", "NA", "NONE", "NULL"}:
                value = ""
            output.append(value)
    return output or [""]


def build_answer_map(
    rows: Iterable[Dict[str, Any]],
    split_answers: bool,
    keep_leading_labels: bool,
) -> Dict[str, List[str]]:
    answer_map: Dict[str, List[str]] = {}
    for index, row in enumerate(rows):
        question_id = str(row.get("question_id", row.get("id", ""))).strip()
        if not question_id:
            raise ValueError(f"Prediction row {index} has no question_id/id.")
        if question_id in answer_map:
            raise ValueError(f"Duplicate prediction id: {question_id}")
        answer_map[question_id] = extract_answers(row, split_answers, keep_leading_labels)
    return answer_map


def main() -> None:
    args = parse_args()
    predictions = load_json(Path(args.predictions))
    answer_map = build_answer_map(predictions, args.split_answers, args.keep_leading_labels)
    dataset_rows = fetch_dataset_rows(args.dataset, args.config, args.split, args.page_size)

    submission: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in dataset_rows:
        question_id = str(row["question_id"])
        seen_ids.add(question_id)
        answers = answer_map.get(question_id, [""])
        submission.append(
            {
                "question_id": question_id,
                "answers": answers,
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
