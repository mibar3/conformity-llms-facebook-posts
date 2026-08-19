"""
Correct-vs-correct control condition: both posts show the correct claim, only the
engagement scale differs (experiments/e1/*/outputs/e1_results_{condition}_correct_vs_correct_paired.json).
Isolates pure engagement-following behaviour with no correctness signal available to
weigh it against -- a "how automatically does this model follow the crowd" baseline.

Two things computed here:

1. Aggregate chose_higher_engagement rate per model x condition (metrics, likes_only_noise),
   ties excluded (correct_scale == incorrect_scale cells are trivially "correct" and would
   inflate the rate without being informative -- see docs/SESSION_HANDOFF.md 2026-08-05).

2. A directed test on a pattern noticed in the aggregate numbers: Gemma-12B -- the one model
   with independent evidence of "knowing" the correct answer (see gee_analysis.ipynb's
   opposite-corner/diagonal tests) -- has a visibly lower rate under `metrics` (91.9%) than
   under `likes_only_noise` (99.1%), while the other three models are near-ceiling in both.
   Two per-image paired exact sign tests confirm this is a real, image-consistent effect
   rather than an aggregate artifact:
     (a) within Gemma-12B: metrics rate vs. its own likes_only_noise rate, per image
     (b) Gemma-12B's metrics rate vs. the pooled other-3-models' metrics rate, per image

Pure stdlib (no pandas/scipy), same reasoning and same exact sign test as the rest of
statistical_analysis/ -- this checkout has never had pip available.

Run: python3 statistical_analysis/correct_vs_correct_analysis.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS = {
    "gemma-12b":   ROOT_DIR / "experiments/e1/gemma4-12b/outputs",
    "gemma-e4b":   ROOT_DIR / "experiments/e1/gemma4-e4b/outputs",
    "qwen3-vl-4b": ROOT_DIR / "experiments/e1/qwen3-vl-4b/outputs",
    "qwen3-vl-8b": ROOT_DIR / "experiments/e1/qwen3-vl-8b/outputs",
    # Added 2026-08-19: confirmed main roster is six models -- Ministral-3-8B/3-14B replace
    # Pixtral-12B and Mistral Small 3.1 24B (now Appendix E supplementary models).
    "ministral-3-8b":  ROOT_DIR / "experiments/e1/ministral-3-8b/outputs",
    "ministral-3-14b": ROOT_DIR / "experiments/e1/ministral-3-14b/outputs",
}
CONDITIONS = ["metrics", "likes_only_noise"]

OUT_DIR = ROOT_DIR / "statistical_analysis/outputs"


def sign_test_p(n, k):
    """Exact two-sided sign test p-value (binomial, p=0.5)."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_records(model, condition):
    path = MODELS[model] / f"e1_results_{condition}_correct_vs_correct_paired.json"
    return json.loads(path.read_text())


def aggregate_rate(records):
    rows = [r for r in records if r["scale_a"] != r["scale_b"]]
    n = len(rows)
    k = sum(1 for r in rows if r["liked_higher_engagement"])
    return n, k, 100 * k / n


def per_image_rate(records):
    grp = defaultdict(list)
    for r in records:
        if r["scale_a"] == r["scale_b"]:
            continue
        grp[r["num"]].append(1 if r["liked_higher_engagement"] else 0)
    return {num: sum(v) / len(v) for num, v in grp.items()}


def paired_sign_test(rates_a, rates_b, label):
    nums = sorted(set(rates_a) & set(rates_b))
    diffs = [rates_a[num] - rates_b[num] for num in nums]
    neg = sum(1 for d in diffs if d < 0)
    pos = sum(1 for d in diffs if d > 0)
    tied = sum(1 for d in diffs if d == 0)
    n_decisive = neg + pos
    p = sign_test_p(n_decisive, neg)
    print(f"[{label}] n_images={len(nums)} a<b={neg} a>b={pos} tied={tied} p={p:.3e}")
    return {
        "comparison": label,
        "n_images": len(nums),
        "n_a_lower": neg,
        "n_a_higher": pos,
        "n_tied": tied,
        "p_value": p,
    }


def main():
    summary_rows = []
    print("=" * 90)
    print("Aggregate chose_higher_engagement rate (ties excluded)")
    print("=" * 90)
    for model in MODELS:
        for condition in CONDITIONS:
            records = load_records(model, condition)
            n, k, pct = aggregate_rate(records)
            summary_rows.append({"model": model, "condition": condition, "n": n, "n_chose_higher": k, "pct_chose_higher": pct})
            print(f"{model:13s} {condition:18s} n={n:5d} chose_higher_engagement={pct:5.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "correct_vs_correct_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 90)
    print("Directed tests: Gemma-12B's metrics gap")
    print("=" * 90)
    gemma_metrics = per_image_rate(load_records("gemma-12b", "metrics"))
    gemma_noise = per_image_rate(load_records("gemma-12b", "likes_only_noise"))

    other_metrics_by_model = {
        model: per_image_rate(load_records(model, "metrics"))
        for model in MODELS if model != "gemma-12b"
    }
    nums = sorted(set.intersection(*[set(r) for r in other_metrics_by_model.values()]))
    pooled_other_metrics = {
        num: sum(other_metrics_by_model[m][num] for m in other_metrics_by_model) / len(other_metrics_by_model)
        for num in nums
    }

    test_rows = [
        paired_sign_test(gemma_metrics, gemma_noise, "gemma-12b: metrics < own likes_only_noise"),
        paired_sign_test(gemma_metrics, pooled_other_metrics, f"gemma-12b: metrics < pooled other-{len(other_metrics_by_model)}-models metrics"),
    ]

    with open(OUT_DIR / "correct_vs_correct_gemma12b_metrics_gap_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"\nSaved to {OUT_DIR}/correct_vs_correct_summary.csv "
          f"and {OUT_DIR}/correct_vs_correct_gemma12b_metrics_gap_test.csv")


if __name__ == "__main__":
    main()
