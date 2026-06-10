"""
OpenQA Ensemble Script
======================
Combines predictions from multiple VLMs using position-aware weighted voting.

Models and their scores:
  1. qwen3_32b   : 0.6270  â†’ weight 4.0
  2. qwen3_8b    : 0.6093  â†’ weight 3.0
  3. qwen25_32b  : 0.6125  â†’ weight 3.0
  4. qwen36_35b  : 0.5663  â†’ weight 2.0
  5. internvl3_8b: 0.4681  â†’ weight 1.0

Strategies implemented:
  A) position_weighted  â€“ position-aware weighted voting (main strategy)
  B) top2_union         â€“ qwen3_32b backbone enriched with qwen3_8b alternatives
  C) weighted_union     â€“ all unique answers, ordered by model weight
  D) qwen3_32b_only     â€“ baseline (best single model)
"""

import json
import re
import argparse
from collections import Counter
from pathlib import Path

# â”€â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

FILES = {
    "qwen3_32b":    "/mnt/user-data/uploads/openqa_qwen3vl32b_thinking_test_official.json",
    "qwen3_8b":     "/mnt/user-data/uploads/openqa_qwen3vl8b_direct_test_official.json",
    "qwen25_32b":   "/mnt/user-data/uploads/openqa_qwen25vl32b_bnb4bit_test_official.json",
    "qwen36_35b":   "/mnt/user-data/uploads/openqa_qwen36_35b_a3b_base_test_official.json",
    "internvl3_8b": "/mnt/user-data/uploads/openqa_internvl3_8b_hf_test_rerun_official.json",
}

# Scores â†’ normalized weights (softmax-like scaling)
MODEL_SCORES = {
    "qwen3_32b":    0.6270,
    "qwen3_8b":     0.6093,
    "qwen25_32b":   0.6125,
    "qwen36_35b":   0.5663,
    "internvl3_8b": 0.4681,
}

# Exponential weights amplify score differences
import math
MODEL_WEIGHTS = {k: math.exp(v * 10) for k, v in MODEL_SCORES.items()}

# Normalise to sum=1
_total = sum(MODEL_WEIGHTS.values())
MODEL_WEIGHTS = {k: v / _total for k, v in MODEL_WEIGHTS.items()}

# â”€â”€â”€ Utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def clean_answer(ans: str) -> str:
    """Remove model artifacts and normalise whitespace."""
    ans = re.sub(r"<\|im_end\|>", "", ans)
    ans = re.sub(r"<\|.*?\|>", "", ans)        # any leftover special tokens
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def dedup_halved(answers: list[str]) -> list[str]:
    """Remove exact halved duplications (qwen36_35b artifact)."""
    n = len(answers)
    if n >= 2 and n % 2 == 0 and answers[: n // 2] == answers[n // 2 :]:
        return answers[: n // 2]
    return answers


def normalize(text: str) -> str:
    return re.sub(r"[\s\W]+", " ", text.lower()).strip()


def token_set(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_all() -> dict[str, dict[str, dict]]:
    data: dict[str, dict[str, dict]] = {}
    for name, path in FILES.items():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        data[name] = {d["question_id"]: d for d in raw}
    return data


def get_clean_answers(data: dict, name: str, qid: str) -> list[str]:
    entry = data[name].get(qid)
    if not entry:
        return []
    answers = dedup_halved(entry["answers"])
    cleaned = [clean_answer(a) for a in answers]
    return [a for a in cleaned if a]  # drop empty strings


# â”€â”€â”€ Strategy A: Position-Aware Weighted Voting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def strategy_position_weighted(data: dict, qid: str) -> list[str]:
    """
    For each expected answer slot:
      1. Collect one answer per model (if available at that position).
      2. Cluster by Jaccard similarity (threshold=0.35).
      3. Within each cluster, the representative is the answer from the
         highest-weight model in that cluster.
      4. Pick the cluster with the highest summed weight.
    The number of answer slots is chosen by weighted-majority vote.
    """
    all_answers: dict[str, list[str]] = {
        name: get_clean_answers(data, name, qid) for name in FILES
    }

    # --- Determine target answer count via weighted vote ---
    count_vote: Counter = Counter()
    for name, answers in all_answers.items():
        n = len(answers)
        if n > 0:
            count_vote[n] += MODEL_WEIGHTS[name]

    if not count_vote:
        return [""]

    # Prefer the count from the top-2 models when they agree
    top2_models = ["qwen3_32b", "qwen3_8b"]
    top2_counts = [len(all_answers[m]) for m in top2_models if all_answers[m]]
    if len(top2_counts) == 2 and top2_counts[0] == top2_counts[1]:
        target_n = top2_counts[0]
    else:
        target_n = count_vote.most_common(1)[0][0]

    # --- For each position, select the best answer ---
    ensemble: list[str] = []
    for i in range(target_n):
        candidates: list[tuple[str, float]] = []
        for name, answers in all_answers.items():
            if i < len(answers):
                candidates.append((answers[i], MODEL_WEIGHTS[name]))

        if not candidates:
            break

        # Single candidate â†’ take it
        if len(candidates) == 1:
            ensemble.append(candidates[0][0])
            continue

        # Cluster similar answers
        # Each cluster: (representative, best_model_weight, total_weight)
        clusters: list[tuple[str, float, float]] = []
        for ans, weight in candidates:
            placed = False
            for idx, (rep, best_w, total_w) in enumerate(clusters):
                if jaccard(ans, rep) >= 0.35:
                    if weight > best_w:
                        # Higher-weight model â†’ promote its answer as rep
                        clusters[idx] = (ans, weight, total_w + weight)
                    else:
                        clusters[idx] = (rep, best_w, total_w + weight)
                    placed = True
                    break
            if not placed:
                clusters.append((ans, weight, weight))

        # Pick the cluster with the highest total weight
        best_rep, _, _ = max(clusters, key=lambda x: x[2])
        ensemble.append(best_rep)

    return ensemble if ensemble else [""]


# â”€â”€â”€ Strategy B: Top-2 Union â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def strategy_top2_union(data: dict, qid: str) -> list[str]:
    """
    Use qwen3_32b as the backbone.
    For each position, if qwen3_8b's answer is substantially different
    (Jaccard < 0.3) AND qwen25_32b also agrees with qwen3_8b, prefer qwen3_8b.
    """
    primary   = get_clean_answers(data, "qwen3_32b", qid)
    secondary = get_clean_answers(data, "qwen3_8b",   qid)
    tertiary  = get_clean_answers(data, "qwen25_32b", qid)

    if not primary:
        return secondary or [""]

    target_n = len(primary)
    ensemble: list[str] = []
    for i in range(target_n):
        a_primary = primary[i] if i < len(primary) else ""
        a_secondary = secondary[i] if i < len(secondary) else ""
        a_tertiary  = tertiary[i]  if i < len(tertiary)  else ""

        if not a_secondary:
            ensemble.append(a_primary)
            continue

        j_ps = jaccard(a_primary, a_secondary)
        if j_ps < 0.3:
            # secondary differs â†’ check if tertiary supports secondary
            j_st = jaccard(a_secondary, a_tertiary)
            if j_st >= 0.3:
                ensemble.append(a_secondary)  # two models agree against primary
            else:
                ensemble.append(a_primary)    # trust primary
        else:
            ensemble.append(a_primary)        # both agree, keep primary

    return ensemble if ensemble else [""]


# â”€â”€â”€ Strategy C: Weighted Union â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def strategy_weighted_union(data: dict, qid: str) -> list[str]:
    """
    Pool ALL unique answers from all models.
    Score each unique answer by the sum of weights of models that produced it
    (or something similar to it at any position).
    Return the top-N unique answers sorted by score, where N is determined
    by the weighted-majority-vote on answer count.
    """
    all_answers: dict[str, list[str]] = {
        name: get_clean_answers(data, name, qid) for name in FILES
    }

    count_vote: Counter = Counter()
    for name, answers in all_answers.items():
        n = len(answers)
        if n > 0:
            count_vote[n] += MODEL_WEIGHTS[name]

    target_n = count_vote.most_common(1)[0][0] if count_vote else 1

    # Score pool: unique answer â†’ cumulative weight
    scored: list[tuple[str, float]] = []
    for name, answers in all_answers.items():
        for ans in answers:
            placed = False
            for idx, (rep, w) in enumerate(scored):
                if jaccard(ans, rep) >= 0.4:
                    best_rep = rep if len(rep) >= len(ans) else ans
                    scored[idx] = (best_rep, w + MODEL_WEIGHTS[name])
                    placed = True
                    break
            if not placed:
                scored.append((ans, MODEL_WEIGHTS[name]))

    scored.sort(key=lambda x: -x[1])
    top = [a for a, _ in scored[:target_n]]
    return top if top else [""]


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_ensemble(strategy: str, out_path: str) -> None:
    print(f"Loading data â€¦")
    data = load_all()
    all_qids = list(data["qwen3_32b"].keys())
    print(f"  {len(all_qids)} questions across 6 languages\n")

    fn_map = {
        "position_weighted": strategy_position_weighted,
        "top2_union":        strategy_top2_union,
        "weighted_union":    strategy_weighted_union,
        "qwen3_32b_only":    lambda d, q: get_clean_answers(d, "qwen3_32b", q),
    }
    fn = fn_map[strategy]

    results = []
    count_stats = Counter()
    for qid in all_qids:
        answers = fn(data, qid)
        language = (
            data["qwen3_32b"].get(qid, {}).get("language")
            or next(
                (data[m][qid]["language"] for m in FILES if qid in data[m]),
                "English",
            )
        )
        results.append({"question_id": qid, "answers": answers, "language": language})
        count_stats[len(answers)] += 1

    print(f"Answer-count distribution: {dict(sorted(count_stats.items()))}")
    print(f"Writing {len(results)} entries â†’ {out_path}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Done âœ“")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["position_weighted", "top2_union", "weighted_union", "qwen3_32b_only"],
        default="position_weighted",
    )
    parser.add_argument("--out", default="/mnt/user-data/outputs/ensemble_openqa.json")
    args = parser.parse_args()
    run_ensemble(args.strategy, args.out)

