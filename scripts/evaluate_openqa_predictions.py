import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset


LANGUAGE_CODES = {
    "Bulgarian": "bg",
    "Chinese": "zh",
    "Croatian": "hr",
    "English": "en",
    "Italian": "it",
    "Serbian": "sr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Visual OpenQA predictions with BLEU, ROUGE-L, METEOR, and optional COMET.")
    parser.add_argument("predictions", help="Prediction JSON or raw JSONL with question_id and answer/answers.")
    parser.add_argument("--dataset", default="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None)
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--id-column", default="question_id")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--source-column", default="auto", help="Optional source text for COMET. Use auto, none, or a dataset column.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--comet", action="store_true", help="Compute COMET if unbabel-comet is installed.")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-batch-size", type=int, default=8)
    parser.add_argument("--round-digits", type=int, default=4)
    return parser.parse_args()


def load_predictions(path: Path) -> Dict[str, str]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL rows.")

    result: Dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} row {index} is not an object.")
        question_id = str(row.get("question_id", row.get("id", ""))).strip()
        if not question_id:
            raise ValueError(f"{path} row {index} has no question_id.")
        if "answer" in row:
            answer = str(row.get("answer", "")).strip()
        elif "answers" in row and isinstance(row["answers"], list):
            answer = str(row["answers"][0] if row["answers"] else "").strip()
        else:
            answer = str(row.get("prediction", "")).strip()
        result[question_id] = answer
    return result


def normalize_text(text: Any) -> str:
    value = str(text)
    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"</?think>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:final\s+answer|answer)\s*(?:is|:|-)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    if re.search(r"[\u4e00-\u9fff]", text):
        chars = [char for char in text if not char.isspace()]
        return chars
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def ngram_counts(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))


def sentence_bleu(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        pred_counts = ngram_counts(pred_tokens, n)
        ref_counts = ngram_counts(ref_tokens, n)
        total = sum(pred_counts.values())
        if total == 0:
            precisions.append(1e-9)
            continue
        overlap = sum((pred_counts & ref_counts).values())
        precisions.append((overlap + 1.0) / (total + 1.0))

    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1.0 - len(ref_tokens) / max(1, len(pred_tokens)))
    return brevity * math.exp(sum(math.log(max(precision, 1e-9)) for precision in precisions) / 4.0)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def meteor_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    matches = sum(common.values())
    if matches == 0:
        return 0.0
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (10 * precision * recall) / (recall + 9 * precision)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def rounded(value: float, digits: int) -> float:
    return round(float(value), digits)


def pick_source(row: Dict[str, Any], source_column: str) -> str:
    if source_column == "none":
        return ""
    if source_column != "auto":
        return str(row.get(source_column, ""))
    for candidate in ("question", "question_text", "text", "prompt", "ocr", "caption"):
        if candidate in row:
            return str(row[candidate])
    return ""


def compute_comet(rows: List[Dict[str, Any]], model_name: str, batch_size: int) -> List[float]:
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    comet_rows = [
        {
            "src": row.get("source", ""),
            "mt": row["prediction"],
            "ref": row["reference"],
        }
        for row in rows
    ]
    output = model.predict(comet_rows, batch_size=batch_size, gpus=1)
    return [float(score) for score in output.scores]


def main() -> None:
    args = parse_args()
    predictions = load_predictions(Path(args.predictions))
    dataset_kwargs: Dict[str, Any] = {"path": args.dataset, "split": args.split}
    if args.config:
        dataset_kwargs["name"] = args.config
    dataset = load_dataset(**dataset_kwargs)

    rows: List[Dict[str, Any]] = []
    for row in dataset:
        question_id = str(row[args.id_column])
        gold = str(row.get(args.answer_column, "")).strip()
        prediction = predictions.get(question_id, "").strip()
        language = str(row.get(args.language_column, "Unknown"))
        if not gold or gold.upper() == "HIDDEN":
            continue
        rows.append(
            {
                "question_id": question_id,
                "prediction": normalize_text(prediction),
                "reference": normalize_text(gold),
                "language": language,
                "source": pick_source(row, args.source_column),
            }
        )

    if not rows:
        raise ValueError("No scorable rows found.")

    for row in rows:
        row["bleu"] = sentence_bleu(row["prediction"], row["reference"])
        row["rouge_l"] = rouge_l(row["prediction"], row["reference"])
        row["meteor"] = meteor_score(row["prediction"], row["reference"])

    if args.comet:
        try:
            comet_scores = compute_comet(rows, args.comet_model, args.comet_batch_size)
            for row, score in zip(rows, comet_scores):
                row["comet"] = score
        except Exception as exc:
            print(f"Warning: could not compute COMET: {exc}")

    by_language: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)

    metrics: Dict[str, Any] = {
        "num_samples": len(rows),
        "bleu_avg": rounded(mean(row["bleu"] for row in rows), args.round_digits),
        "rouge_l": rounded(mean(row["rouge_l"] for row in rows), args.round_digits),
        "meteor": rounded(mean(row["meteor"] for row in rows), args.round_digits),
    }

    if "comet" in rows[0]:
        metrics["comet_overall"] = rounded(mean(row["comet"] for row in rows), args.round_digits)
        for language, language_rows in sorted(by_language.items()):
            code = LANGUAGE_CODES.get(language, language.lower()[:2])
            metrics[f"comet_{code}"] = rounded(mean(row["comet"] for row in language_rows), args.round_digits)

    per_language: Dict[str, Dict[str, float]] = {}
    for language, language_rows in sorted(by_language.items()):
        per_language[language] = {
            "bleu_avg": rounded(mean(row["bleu"] for row in language_rows), args.round_digits),
            "rouge_l": rounded(mean(row["rouge_l"] for row in language_rows), args.round_digits),
            "meteor": rounded(mean(row["meteor"] for row in language_rows), args.round_digits),
        }
    metrics["per_language"] = per_language

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics: {output_path}")


if __name__ == "__main__":
    main()
