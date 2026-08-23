"""
Two grid-level tests, suggested by the supervisor (2026-07-16 feedback items 2-3,
see docs/SESSION_HANDOFF.md), applied to the simple-plot pilot models
(Gemma-E4B and Qwen3-VL-8B, experiments/e1_simple_plot/) and compared against the
same tests already run on the original-chart models via statistical_analysis/gee_analysis.ipynb
(statistical_analysis/outputs/opposite_corner_test.csv, diagonal_above_chance_test.csv).

1. Opposite-corner test (rewritten 2026-08, supervisor correction -- the original sum-based
   version below is wrong): for a grid cell (correct_scale=X, incorrect_scale=Y) with X<Y --
   the "disadvantaged" cell, correct post has less engagement -- and its mirror
   (correct_scale=Y, incorrect_scale=X) -- the "advantaged" cell -- compute the % of images
   choosing the correct post in each (aggregated across images, i.e. the same per-cell
   percentage the 7x7 grid heatmaps already show), then take
   ratio = disadvantaged_pct / advantaged_pct.
   Ratio == 1 means engagement has no effect on the decision at all (the model does equally
   well/poorly regardless of which post has more engagement) -- no conformity. Ratio -> 0
   means engagement fully determines the outcome (never picks correct when it's disadvantaged,
   always when it's advantaged) -- a strong conformity/engagement effect. The disadvantaged
   cell is used as the numerator specifically because it is the one that can legitimately hit
   0% (full conformity collapse); the advantaged cell essentially never does, so division is
   safe -- but both are given a Haldane-Anscombe continuity correction (+0.5/+1 on the
   underlying counts) regardless, so a literal 0% disadvantaged cell still yields a finite,
   comparable ratio instead of exactly 0. The single most extreme pair (correct=0 vs
   incorrect=1,000,000, i.e. the actual top-left/bottom-right corners of the 7x7 grid as
   plotted) is reported separately as the headline number; all 21 off-diagonal mirrored pairs
   are also reported for the full picture.

   [Previous version, kept here for the record only, not used below: summed the two mirrored
   cells' 0/1 per-image outcomes and sign-tested whether the sum exceeded 1 across images --
   intended to test for a correctness effect independent of engagement, not to size the
   engagement effect itself, which is what was actually wanted.]

2. Diagonal-above-chance test: per-image accuracy at disparity=0 (correct_scale ==
   incorrect_scale, no engagement pressure either way), tested against chance (50%) with
   an exact sign test. This is the model's "competence" on the task with no social-proof
   signal present.

Pure stdlib (no pandas/scipy) -- same reasoning as logprob_analysis.py: this checkout has
never had pip available, and an exact binomial sign test needs nothing else.

Run: python3 statistical_analysis/simple_plot_grid_tests.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS = {
    "gemma4-e4b-simple-plot":  ROOT_DIR / "experiments/e1_simple_plot/gemma4-e4b/outputs",
    "qwen3-vl-8b-simple-plot": ROOT_DIR / "experiments/e1_simple_plot/qwen3-vl-8b/outputs",
}
CONDITIONS = ["metrics", "likes_only_noise"]
SCALES = [0, 10, 100, 1000, 10000, 100000, 1000000]

OUT_DIR = ROOT_DIR / "statistical_analysis/outputs"


def sign_test_p(n, k):
    """Exact two-sided sign test p-value (binomial, p=0.5). Same method used
    throughout this project's other exact tests (see docs/THESIS.md §6.1)."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_grid(path):
    records = json.loads(path.read_text())
    lookup = {}
    images = set()
    for r in records:
        chose_correct = 1 if r["liked_variant"] == "correct" else 0
        lookup[(r["num"], r["correct_scale"], r["incorrect_scale"])] = chose_correct
        images.add(r["num"])
    return lookup, images


def cell_pct(lookup, images, cs, ics):
    """Aggregate % of images choosing the correct post at grid cell (correct_scale=cs,
    incorrect_scale=ics), plus the Haldane-Anscombe-corrected version ((k+0.5)/(n+1)) used
    for the ratio so a literal 0% or 100% cell never breaks the division."""
    vals = [lookup[(num, cs, ics)] for num in images if (num, cs, ics) in lookup]
    n = len(vals)
    if n == 0:
        return None
    k = sum(vals)
    return {"n": n, "k": k, "pct": 100 * k / n, "pct_corrected": 100 * (k + 0.5) / (n + 1)}


def opposite_corner_ratio_test(lookup, images):
    """One row per off-diagonal mirrored pair (X<Y): ratio = disadvantaged_pct / advantaged_pct
    (both continuity-corrected). See module docstring for why this replaced the sum-based test."""
    rows = []
    for cs in SCALES:
        for ics in SCALES:
            if cs >= ics:
                continue  # only X<Y, so "disadvantaged" (cs=X) is unambiguous
            disadv = cell_pct(lookup, images, cs, ics)   # correct=X (small), incorrect=Y (large)
            adv = cell_pct(lookup, images, ics, cs)      # correct=Y (large), incorrect=X (small)
            if disadv is None or adv is None:
                continue
            ratio = disadv["pct_corrected"] / adv["pct_corrected"]
            rows.append({
                "correct_scale_disadv": cs,
                "incorrect_scale_disadv": ics,
                "disadvantaged_pct": disadv["pct"],
                "advantaged_pct": adv["pct"],
                "ratio": ratio,
                "n_disadv": disadv["n"],
                "n_adv": adv["n"],
            })
    if not rows:
        return None, None
    extreme = next((r for r in rows if r["correct_scale_disadv"] == SCALES[0]
                     and r["incorrect_scale_disadv"] == SCALES[-1]), None)
    return rows, extreme


def diagonal_test(lookup, images):
    per_image = defaultdict(list)
    for (num, cs, ics), chose_correct in lookup.items():
        if cs == ics:
            per_image[num].append(chose_correct)
    means = {num: sum(v) / len(v) for num, v in per_image.items() if v}
    wins = sum(1 for m in means.values() if m > 0.5)
    losses = sum(1 for m in means.values() if m < 0.5)
    ties = sum(1 for m in means.values() if m == 0.5)
    n = wins + losses
    if not means:
        return None
    p = sign_test_p(n, wins)
    return {
        "mean_diagonal_pct": 100 * sum(means.values()) / len(means),
        "n_images_above_50": wins,
        "n_images_below_50": losses,
        "n_images_at_50": ties,
        "p_value": p,
    }


def main():
    corner_rows = []
    extreme_rows = []
    diagonal_rows = []

    for model, outdir in MODELS.items():
        for condition in CONDITIONS:
            path = outdir / f"e1_results_{condition}_paired.json"
            if not path.exists():
                print(f"{model} / {condition}: missing {path.name}, skipping")
                continue
            lookup, images = load_grid(path)

            pair_rows, extreme = opposite_corner_ratio_test(lookup, images)
            if pair_rows:
                for r in pair_rows:
                    corner_rows.append({"model": model, "signal_type": condition, **r})
                mean_ratio = sum(r["ratio"] for r in pair_rows) / len(pair_rows)
                print(f"[opposite-corner] {model:26s} {condition:18s} "
                      f"mean_ratio(21 pairs)={mean_ratio:6.3f}  "
                      f"extreme(0v1M)_ratio={extreme['ratio']:6.3f} "
                      f"({extreme['disadvantaged_pct']:.1f}% / {extreme['advantaged_pct']:.1f}%)"
                      if extreme else
                      f"[opposite-corner] {model:26s} {condition:18s} mean_ratio(21 pairs)={mean_ratio:6.3f}")
            if extreme:
                extreme_rows.append({"model": model, "signal_type": condition, **extreme})

            diagonal = diagonal_test(lookup, images)
            if diagonal:
                row = {"model": model, "signal_type": condition, **diagonal}
                diagonal_rows.append(row)
                print(f"[diagonal]        {model:26s} {condition:18s} "
                      f"mean_diagonal={diagonal['mean_diagonal_pct']:6.2f}%  "
                      f"above50={diagonal['n_images_above_50']:3d} "
                      f"below50={diagonal['n_images_below_50']:3d} "
                      f"tie={diagonal['n_images_at_50']:3d}  p={diagonal['p_value']:.3e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "simple_plot_opposite_corner_ratio_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(corner_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corner_rows)

    with open(OUT_DIR / "simple_plot_opposite_corner_ratio_extreme.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(extreme_rows[0].keys()))
        writer.writeheader()
        writer.writerows(extreme_rows)

    with open(OUT_DIR / "simple_plot_diagonal_above_chance_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagonal_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagonal_rows)

    print(f"\nSaved to {OUT_DIR}/simple_plot_opposite_corner_ratio_test.csv, "
          f"{OUT_DIR}/simple_plot_opposite_corner_ratio_extreme.csv, "
          f"and {OUT_DIR}/simple_plot_diagonal_above_chance_test.csv")


if __name__ == "__main__":
    main()
