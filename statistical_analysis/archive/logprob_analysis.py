"""
Descriptive analysis of the isolated-judgment log-probability data (see
experiments/e1/e1_utils/logprob_scoring.py) across the models that currently have it:
Gemma-12B, Gemma-E4B, Qwen3-VL-4B, Qwen3-VL-8B (Pixtral-12B and Mistral Small 3.1 24B
pending the roster-exclusion decision, see docs/SESSION_HANDOFF.md).

Pure stdlib (no pandas/numpy/scipy) so it runs anywhere Python 3 runs, including this
project's local dev checkout, which has never had pip available. Spearman correlation is
implemented from scratch (rank + Pearson) for the same reason. Stays at the descriptive
layer only (no significance tests) by explicit decision, 2026-08-03 — see chat/handoff for
what to add if that changes (a paired sign test akin to the diagonal-vs-chance test in
gee_analysis.ipynb would be the natural next step, once there's a specific hypothesis to
confirm rather than explore).

Run: python3 statistical_analysis/logprob_analysis.py
"""

import json
import math
import statistics
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Confirmed six-model roster. The two Ministral entries were added 2026-08-23: this script had
# been left on the pre-Ministral four-model roster, so every log-probability figure in the thesis
# was a four-model result reported on a six-model roster. Their single-image `likes_only_noise`
# runs did not exist until the same day either -- the Ministral notebooks descend from the
# Mistral-Small-24B template, which never carried those cells.
MODELS = {
    "gemma-12b":       ROOT_DIR / "experiments/e1/gemma4-12b/outputs",
    "gemma-e4b":       ROOT_DIR / "experiments/e1/gemma4-e4b/outputs",
    "qwen3-vl-4b":     ROOT_DIR / "experiments/e1/qwen3-vl-4b/outputs",
    "qwen3-vl-8b":     ROOT_DIR / "experiments/e1/qwen3-vl-8b/outputs",
    "ministral-3-8b":  ROOT_DIR / "experiments/e1/ministral-3-8b/outputs",
    "ministral-3-14b": ROOT_DIR / "experiments/e1/ministral-3-14b/outputs",
}

# `likes_only` was dropped 2026-08-23. It is the predecessor of `likes_only_noise` -- superseded
# once round-number anchoring was controlled for -- and is not reported anywhere in the thesis,
# which covers baseline, likes_only_noise and realistic metrics only. Its raw files have been
# removed for Ministral-3-14B, so keeping the condition here would break the six-model run for a
# figure nothing cites.
CONDITIONS = {
    "baseline":         {"single": "e1_results_baseline_logprobs.json",         "yesno": "e1_results_baseline_yesno_logprobs.json"},
    "metrics":          {"single": "e1_results_metrics_logprobs.json",          "yesno": "e1_results_metrics_yesno_logprobs.json"},
    "likes_only_noise": {"single": "e1_results_likes_only_noise_logprobs.json", "yesno": "e1_results_likes_only_noise_yesno_logprobs.json"},
}
WORD = {"single": "like", "yesno": "yes"}


def load_rows(path, like_word):
    records = json.loads(path.read_text())
    rows = []
    for r in records:
        rows.append({
            "image": r["image"], "variant": r["variant"],
            "scale_value": r.get("scale_value"),
            "decoded_answer": r["answer"],
            "p_like": r["candidates"][like_word]["prob_forced_choice"],
        })
    return rows


def rankdata(values):
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    varx = sum((xi - mx) ** 2 for xi in x)
    vary = sum((yi - my) ** 2 for yi in y)
    if varx == 0 or vary == 0:
        return float("nan")
    return cov / math.sqrt(varx * vary)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def mean(vals):
    return sum(vals) / len(vals)


def stdev(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def load_all_rows():
    all_rows = []
    for model, outdir in MODELS.items():
        for condition, files in CONDITIONS.items():
            for phrasing, like_word in WORD.items():
                fpath = outdir / files[phrasing]
                if not fpath.exists():
                    continue  # lets this run today even before every model/condition lands
                rows = load_rows(fpath, like_word)
                for r in rows:
                    r["model"] = model
                    r["condition"] = condition
                    r["phrasing"] = phrasing
                    r["decoded_like_binary"] = 1 if r["decoded_answer"] in ("like", "yes") else 0
                all_rows.extend(rows)
    return all_rows


def section2_saturation_check(all_rows):
    print("=" * 100)
    print("SECTION 2 — Saturation check: decoded rate vs. underlying p_like spread")
    print("=" * 100)
    groups = {}
    for r in all_rows:
        key = (r["model"], r["condition"], r["phrasing"])
        groups.setdefault(key, []).append(r)

    header = f"{'model':13s} {'condition':18s} {'phrasing':8s} {'n':>6s} {'decoded_%':>10s} {'mean_p_like':>12s} {'std_p_like':>11s} {'%confident':>11s}"
    print(header)
    print("-" * len(header))
    summary_rows = []
    for (model, condition, phrasing), rows in sorted(groups.items()):
        p_likes = [r["p_like"] for r in rows]
        decoded = [r["decoded_like_binary"] for r in rows]
        decoded_rate = mean(decoded) * 100
        mp = mean(p_likes)
        sp = stdev(p_likes)
        pct_conf = 100 * sum(1 for p in p_likes if p < 0.01 or p > 0.99) / len(p_likes)
        summary_rows.append((model, condition, phrasing, len(rows), decoded_rate, mp, sp, pct_conf))
        print(f"{model:13s} {condition:18s} {phrasing:8s} {len(rows):6d} {decoded_rate:10.1f} {mp:12.4f} {sp:11.4f} {pct_conf:11.1f}")

    print("\n--- Cells where decoded answer is saturated (>=99% or <=1%) but p_like still shows real spread (std > 0.05) ---")
    flagged = [row for row in summary_rows if (row[4] >= 99 or row[4] <= 1) and row[6] > 0.05]
    if flagged:
        for model, condition, phrasing, n, decoded_rate, mp, sp, pct_conf in flagged:
            print(f"  {model:13s} {condition:18s} {phrasing:8s}  decoded={decoded_rate:.1f}%  mean_p_like={mp:.4f}  std={sp:.4f}")
    else:
        print("  (none — saturation appears to be genuine everywhere it occurs)")


def section3_scale_correlation(all_rows):
    print("\n" + "=" * 100)
    print("SECTION 3 — Does p_like track engagement disparity (scale_value) even where decode is flat?")
    print("=" * 100)
    scale_groups = {}
    for r in all_rows:
        if r["scale_value"] is None:
            continue
        key = (r["model"], r["condition"], r["phrasing"], r["variant"])
        scale_groups.setdefault(key, []).append(r)

    header = f"{'model':13s} {'condition':18s} {'phrasing':8s} {'variant':10s} {'n':>6s} {'spearman_rho':>13s}"
    print(header)
    print("-" * len(header))
    for (model, condition, phrasing, variant), rows in sorted(scale_groups.items()):
        scale_vals = [r["scale_value"] for r in rows]
        if len(set(scale_vals)) < 2:
            continue
        p_likes = [r["p_like"] for r in rows]
        rho = spearman(scale_vals, p_likes)
        print(f"{model:13s} {condition:18s} {phrasing:8s} {variant:10s} {len(rows):6d} {rho:13.3f}")


def section4_gap_comparison(all_rows):
    print("\n" + "=" * 100)
    print("SECTION 4 — Correctness-sensitivity gap: logprob-based vs. decoded-answer-based")
    print("=" * 100)
    gap_groups = {}
    for r in all_rows:
        key = (r["model"], r["condition"], r["phrasing"])
        gap_groups.setdefault(key, {"correct": [], "incorrect": []})
        if r["variant"] in ("correct", "incorrect"):
            gap_groups[key][r["variant"]].append(r)

    header = f"{'model':13s} {'condition':18s} {'phrasing':8s} {'logprob_gap':>12s} {'decoded_gap_%':>14s}"
    print(header)
    print("-" * len(header))
    for (model, condition, phrasing), variants in sorted(gap_groups.items()):
        c_rows, i_rows = variants["correct"], variants["incorrect"]
        if not c_rows or not i_rows:
            continue
        logprob_gap = mean([r["p_like"] for r in c_rows]) - mean([r["p_like"] for r in i_rows])
        decoded_gap = (mean([r["decoded_like_binary"] for r in c_rows]) - mean([r["decoded_like_binary"] for r in i_rows])) * 100
        print(f"{model:13s} {condition:18s} {phrasing:8s} {logprob_gap:12.4f} {decoded_gap:14.1f}")


if __name__ == "__main__":
    all_rows = load_all_rows()
    print(f"Loaded {len(all_rows)} total trials across {len(MODELS)} models.\n")
    section2_saturation_check(all_rows)
    section3_scale_correlation(all_rows)
    section4_gap_comparison(all_rows)
