#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Official-compatible evaluator for ImageCLEF 2026 Visual OpenQA.

This keeps the metric implementation aligned with the official
`2026/evaluate_qa.py` script, while also accepting this repo's legacy
prediction files that use `answer` instead of `answers`.
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    find_dotenv = None
    load_dotenv = None


warnings.filterwarnings("ignore")

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ROUGE_SCORER = None
_BLEU = None
_SINGLE_METEOR_SCORE = None
_NLTK = None
_TQDM = None

_LANGUAGE_ISO2: Dict[str, str] = {
    "bulgarian": "bg",
    "chinese": "zh",
    "croatian": "hr",
    "english": "en",
    "italian": "it",
    "serbian": "sr",
}


def normalise_language(lang: str) -> str:
    """Return the official two-letter language code used in COMET keys."""
    if not lang:
        return "unknown"
    stripped = lang.strip()
    if len(stripped) == 2:
        return stripped.lower()
    return _LANGUAGE_ISO2.get(stripped.lower(), stripped)


if load_dotenv is not None and find_dotenv is not None:
    load_dotenv(find_dotenv(), override=True)


def ensure_metric_dependencies() -> None:
    """Import official metric dependencies only when evaluation starts."""
    global _BLEU, _NLTK, _ROUGE_SCORER, _SINGLE_METEOR_SCORE, _TQDM

    if all(
        dep is not None
        for dep in (_BLEU, _NLTK, _ROUGE_SCORER, _SINGLE_METEOR_SCORE, _TQDM)
    ):
        return

    try:
        import nltk
        from nltk.translate.meteor_score import single_meteor_score
        from rouge_score import rouge_scorer
        from sacrebleu.metrics import BLEU
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing official evaluator dependencies. Install them with: "
            "pip install sacrebleu rouge-score nltk unbabel-comet python-dotenv"
        ) from exc

    _NLTK = nltk
    _SINGLE_METEOR_SCORE = single_meteor_score
    _BLEU = BLEU
    _TQDM = tqdm
    _ROUGE_SCORER = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ImageCLEF 2026 Visual OpenQA predictions with the same "
            "metric formulas as the official 2026/evaluate_qa.py script."
        )
    )
    parser.add_argument(
        "predictions",
        nargs="?",
        help="Prediction JSON/JSONL. Legacy rows with `answer` are accepted.",
    )
    parser.add_argument("--pred_file", "--pred-file", dest="pred_file")
    parser.add_argument("--gold_file", "--gold-file", dest="gold_file")
    parser.add_argument(
        "--dataset",
        default="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual",
        help="Dataset to use for gold rows when --gold_file is not provided.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--id-column", default="question_id")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--batch_size_comet", "--comet-batch-size", type=int, default=64)
    parser.add_argument("--out_file", "--output", dest="out_file", default="scores.json")
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Skip COMET. Do not use this for official-comparable scores.",
    )
    parser.add_argument(
        "--comet",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--write-gold-file",
        help="Optional path to write the official-format gold file used.",
    )
    parser.add_argument(
        "--write-pred-file",
        help="Optional path to write the official-format prediction file used.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def ensure_outdir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def atomic_write_json(path: str, data: Any) -> None:
    ensure_outdir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def avg(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def round_float_values(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {key: round_float_values(value, ndigits) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_float_values(value, ndigits) for value in obj]
    return obj


def _read_list_or_data(path: str, kind: str) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        data = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(source.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"{kind} file must be a list or {{'data': [...]}}")
    if not data:
        raise ValueError(f"{kind} file is empty")
    return data


def _as_answers(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _canonical_gold_item(
    item: Dict[str, Any],
    *,
    id_column: str = "question_id",
    answer_column: str = "answers",
    language_column: str = "language",
) -> Dict[str, Any]:
    if id_column not in item:
        raise ValueError(f"Gold item is missing required field: {id_column}")

    if "answers" in item:
        answers = _as_answers(item["answers"])
    elif answer_column in item:
        answers = _as_answers(item[answer_column])
    else:
        raise ValueError("Gold item is missing required field: answers")

    result = {
        "question_id": str(item[id_column]),
        "answers": answers,
    }
    if language_column in item:
        result["language"] = str(item.get(language_column, "unknown") or "unknown")
    return result


def _canonical_pred_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if "question_id" not in item and "id" not in item:
        raise ValueError("Prediction item is missing required field: question_id")

    if "answers" in item:
        answers = _as_answers(item["answers"])
    elif "answer" in item:
        answers = _as_answers(item["answer"])
    elif "prediction" in item:
        answers = _as_answers(item["prediction"])
    else:
        raise ValueError("Prediction item is missing required field: answers/answer")

    result = {
        "question_id": str(item.get("question_id", item.get("id"))),
        "answers": answers,
    }
    if "language" in item:
        result["language"] = str(item.get("language", "unknown") or "unknown")
    return result


def _build_id_map(items: List[Dict[str, Any]], required_fields: set, kind: str) -> Dict[str, Dict[str, Any]]:
    id_map: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{kind} item at index {idx} is not a dictionary")

        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(
                f"{kind} item at index {idx} is missing required fields: {missing}"
            )

        item_id = str(item["question_id"])
        if item_id in id_map:
            raise ValueError(f"{kind} file contains duplicate id: {item_id}")

        id_map[item_id] = item
    return id_map


def _join_answers(answers: Any) -> str:
    """Join one answer per sub-question exactly as the official script does."""
    if isinstance(answers, list):
        return "\n".join(str(answer) for answer in answers)
    return str(answers)


def _is_hidden_answer(item: Dict[str, Any]) -> bool:
    joined = _join_answers(item.get("answers", "")).strip().upper()
    return joined in {"", "HIDDEN"}


def build_gold_from_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing dependency: datasets. Install it or pass --gold_file."
        ) from exc

    dataset_kwargs: Dict[str, Any] = {"path": args.dataset, "split": args.split}
    if args.config:
        dataset_kwargs["name"] = args.config
    dataset = load_dataset(**dataset_kwargs)

    gold_items = [
        _canonical_gold_item(
            row,
            id_column=args.id_column,
            answer_column=args.answer_column,
            language_column=args.language_column,
        )
        for row in dataset
    ]

    if all(_is_hidden_answer(item) for item in gold_items):
        raise ValueError(
            f"All gold answers are hidden for {args.dataset} [{args.split}]. "
            "Use train/dev or provide an official gold file."
        )
    return gold_items


def load_gold_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.gold_file:
        return [
            _canonical_gold_item(item)
            for item in _read_list_or_data(args.gold_file, "Gold")
        ]
    return build_gold_from_dataset(args)


def load_pred_items(path: str) -> List[Dict[str, Any]]:
    return [_canonical_pred_item(item) for item in _read_list_or_data(path, "Prediction")]


def merge_input_data(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gold_map = _build_id_map(gold_items, {"question_id", "answers"}, "Gold")
    pred_map = _build_id_map(pred_items, {"question_id", "answers"}, "Prediction")

    gold_ids = set(gold_map.keys())
    pred_ids = set(pred_map.keys())

    missing_in_pred = gold_ids - pred_ids
    extra_in_pred = pred_ids - gold_ids

    if missing_in_pred or extra_in_pred:
        messages = []
        if missing_in_pred:
            messages.append(f"Missing prediction ids: {sorted(list(missing_in_pred))[:5]}")
        if extra_in_pred:
            messages.append(f"Unknown prediction ids: {sorted(list(extra_in_pred))[:5]}")
        raise ValueError(" | ".join(messages))

    merged = []
    for item in gold_items:
        item_id = str(item["question_id"])
        merged_item = {
            "question_id": item_id,
            "gold_answers": gold_map[item_id]["answers"],
            "pred_answers": pred_map[item_id]["answers"],
            "gold_answers_str": _join_answers(gold_map[item_id]["answers"]),
            "pred_answers_str": _join_answers(pred_map[item_id]["answers"]),
        }

        if "language" in gold_map[item_id]:
            merged_item["language"] = gold_map[item_id]["language"]
        elif "language" in pred_map[item_id]:
            merged_item["language"] = pred_map[item_id]["language"]

        merged.append(merged_item)
    return merged


def sentence_bleu_n(hyp: str, ref: str, n: int) -> float:
    assert _BLEU is not None
    metric = _BLEU(max_ngram_order=n, effective_order=True)
    return float(metric.sentence_score(hyp, [ref]).score / 100.0)


def ensure_nltk_meteor_resources() -> None:
    assert _NLTK is not None
    try:
        _NLTK.data.find("corpora/wordnet")
    except LookupError:
        _NLTK.download("wordnet", quiet=True)
    try:
        _NLTK.data.find("corpora/omw-1.4")
    except LookupError:
        _NLTK.download("omw-1.4", quiet=True)


def meteor_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def sentence_meteor(hyp: str, ref: str) -> float:
    assert _SINGLE_METEOR_SCORE is not None
    ref_toks = meteor_tokenize(ref)
    hyp_toks = meteor_tokenize(hyp)
    return float(_SINGLE_METEOR_SCORE(ref_toks, hyp_toks))


def sentence_rouge_scores(hyp: str, ref: str) -> Dict[str, float]:
    assert _ROUGE_SCORER is not None
    scores = _ROUGE_SCORER.score(ref, hyp)
    return {
        "rouge-1": float(scores["rouge1"].fmeasure),
        "rouge-2": float(scores["rouge2"].fmeasure),
        "rouge-l": float(scores["rougeL"].fmeasure),
    }


def load_comet_model(model_name: str = "Unbabel/wmt22-comet-da"):
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing dependency: unbabel-comet. Install official evaluator deps with: "
            "pip install sacrebleu rouge-score nltk unbabel-comet python-dotenv"
        ) from exc

    checkpoint = download_model(model_name)
    return load_from_checkpoint(checkpoint)


def comet_scores_batch(
    model: Any,
    srcs: List[str],
    mts: List[str],
    refs: List[str],
    batch_size: int = 64,
) -> List[float]:
    data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, mts, refs)]
    output = model.predict(data, batch_size=batch_size, accelerator="cpu", num_workers=1)
    return [float(score) for score in output.scores]


def _group_by_language(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        language = item.get("language", "unknown") or "unknown"
        groups.setdefault(language, []).append(item)
    return groups


def step_bleu(items: List[Dict[str, Any]]):
    assert _TQDM is not None
    groups = _group_by_language(items)
    lang_scores = {}
    all_b1: List[float] = []
    all_b2: List[float] = []
    all_b3: List[float] = []
    all_b4: List[float] = []

    for lang, subset in groups.items():
        b1: List[float] = []
        b2: List[float] = []
        b3: List[float] = []
        b4: List[float] = []
        for item in _TQDM(subset, desc=f"BLEU-1..4 [{lang}]"):
            hyp, ref = item["pred_answers_str"], item["gold_answers_str"]
            b1.append(sentence_bleu_n(hyp, ref, 1))
            b2.append(sentence_bleu_n(hyp, ref, 2))
            b3.append(sentence_bleu_n(hyp, ref, 3))
            b4.append(sentence_bleu_n(hyp, ref, 4))
        all_b1.extend(b1)
        all_b2.extend(b2)
        all_b3.extend(b3)
        all_b4.extend(b4)
        s1, s2, s3, s4 = avg(b1), avg(b2), avg(b3), avg(b4)
        lang_scores[lang] = {
            "bleu-1": s1,
            "bleu-2": s2,
            "bleu-3": s3,
            "bleu-4": s4,
            "bleu_avg": avg([s1, s2, s3, s4]),
        }

    s1, s2, s3, s4 = avg(all_b1), avg(all_b2), avg(all_b3), avg(all_b4)
    overall = {
        "bleu-1": s1,
        "bleu-2": s2,
        "bleu-3": s3,
        "bleu-4": s4,
        "bleu_avg": avg([s1, s2, s3, s4]),
    }
    return overall, lang_scores


def step_rouge(items: List[Dict[str, Any]]):
    assert _TQDM is not None
    groups = _group_by_language(items)
    lang_scores = {}
    all_r1: List[float] = []
    all_r2: List[float] = []
    all_rl: List[float] = []

    for lang, subset in _TQDM(groups.items(), desc="ROUGE-1/2/L (languages)"):
        r1: List[float] = []
        r2: List[float] = []
        rl: List[float] = []
        for item in subset:
            hyp, ref = item["pred_answers_str"], item["gold_answers_str"]
            scores = sentence_rouge_scores(hyp, ref)
            r1.append(scores["rouge-1"])
            r2.append(scores["rouge-2"])
            rl.append(scores["rouge-l"])
        all_r1.extend(r1)
        all_r2.extend(r2)
        all_rl.extend(rl)
        lang_scores[lang] = {
            "rouge-1": avg(r1),
            "rouge-2": avg(r2),
            "rouge-l": avg(rl),
        }

    overall = {
        "rouge-1": avg(all_r1),
        "rouge-2": avg(all_r2),
        "rouge-l": avg(all_rl),
    }
    return overall, lang_scores


def step_meteor(items: List[Dict[str, Any]]):
    assert _TQDM is not None
    groups = _group_by_language(items)
    lang_scores = {}
    all_scores: List[float] = []

    for lang, subset in _TQDM(groups.items(), desc="METEOR (languages)"):
        scores = [
            sentence_meteor(item["pred_answers_str"], item["gold_answers_str"])
            for item in subset
        ]
        all_scores.extend(scores)
        lang_scores[lang] = avg(scores)
    return avg(all_scores), lang_scores


def step_comet(items: List[Dict[str, Any]], batch_size: int = 64):
    assert _TQDM is not None
    model = load_comet_model("Unbabel/wmt22-comet-da")

    srcs = [item["gold_answers_str"] for item in items]
    mts = [item["pred_answers_str"] for item in items]
    refs = [item["gold_answers_str"] for item in items]

    all_scores: List[float] = []
    for start in _TQDM(range(0, len(items), batch_size), desc="COMET"):
        end = start + batch_size
        all_scores.extend(
            comet_scores_batch(model, srcs[start:end], mts[start:end], refs[start:end], batch_size)
        )

    lang_buckets: Dict[str, List[float]] = {}
    for item, score in zip(items, all_scores):
        language = item.get("language", "unknown") or "unknown"
        lang_buckets.setdefault(language, []).append(score)

    lang_scores = {language: avg(scores) for language, scores in lang_buckets.items()}
    return avg(all_scores), lang_scores


def evaluate_openqa(
    items: List[Dict[str, Any]],
    batch_size_comet: int = 64,
    compute_comet: bool = True,
) -> Dict[str, Any]:
    ensure_metric_dependencies()
    ensure_nltk_meteor_resources()

    bleu_overall, bleu_lang = step_bleu(items)
    rouge_overall, rouge_lang = step_rouge(items)
    meteor_overall, meteor_lang = step_meteor(items)

    comet_section: Dict[str, float] = {}
    if compute_comet:
        comet_overall, comet_lang = step_comet(items, batch_size=batch_size_comet)
        comet_section["comet_overall"] = comet_overall
        for lang, score in sorted(comet_lang.items()):
            comet_section[f"comet_{normalise_language(lang)}"] = score

    langs = sorted(set(list(bleu_lang) + list(rouge_lang) + list(meteor_lang)))
    per_language = {
        lang: {
            "bleu_avg": bleu_lang.get(lang, {}).get("bleu_avg", 0.0),
            "rouge_l": rouge_lang.get(lang, {}).get("rouge-l", 0.0),
            "meteor": meteor_lang.get(lang, 0.0),
        }
        for lang in langs
    }

    report = {
        "num_samples": len(items),
        "bleu_avg": bleu_overall["bleu_avg"],
        "rouge_l": rouge_overall["rouge-l"],
        "meteor": meteor_overall,
        **comet_section,
        "per_language": per_language,
    }
    return round_float_values(report, ndigits=4)


def main() -> None:
    args = parse_args()
    pred_path = args.pred_file or args.predictions
    if not pred_path:
        raise SystemExit("Provide a prediction file as positional argument or --pred_file.")

    gold_items = load_gold_items(args)
    pred_items = load_pred_items(pred_path)
    items = merge_input_data(gold_items, pred_items)

    if args.write_gold_file:
        atomic_write_json(args.write_gold_file, gold_items)
    if args.write_pred_file:
        pred_map = {str(item["question_id"]): item for item in pred_items}
        aligned_preds = [pred_map[str(item["question_id"])] for item in gold_items]
        atomic_write_json(args.write_pred_file, aligned_preds)

    report = evaluate_openqa(
        items,
        batch_size_comet=args.batch_size_comet,
        compute_comet=not args.no_comet,
    )
    atomic_write_json(args.out_file, report)

    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Metrics written to: {os.path.abspath(args.out_file)}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Wrote metrics: {args.out_file}")


if __name__ == "__main__":
    main()
