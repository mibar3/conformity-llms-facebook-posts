"""
Two grid-level tests, suggested by the supervisor (2026-07-16 feedback items 2-3,
see docs/SESSION_HANDOFF.md), applied to the simple-plot pilot models
(Gemma-E4B and Qwen3-VL-8B, experiments/e1_simple_plot/) and compared against the
same tests already run on the original-chart models via statistical_analysis/gee_analysis.ipynb
(statistical_analysis/outputs/opposite_corner_test.csv, diagonal_above_chance_test.csv).

1. Opposite-corner test: for a grid cell (correct_scale=X, incorrect_scale=Y) and its
   mirror (correct_scale=Y, incorrect_scale=X), a model responding to engagement alone
   (no independent correctness sensitivity) should choose the correct post in one cell
   and the incorrect post in the mirrored cell about equally often -- the two outcomes,
   summed as 0/1 indicators, should average to 1 across many such pairs. A sum reliably
   above 1 is evidence of a genuine correctness effect beyond pure engagement-following.

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


def opposite_corner_test(lookup, images):
    wins = losses = ties = 0
    sums = []
    seen_pairs = set()
    for cs in SCALES:
        for ics in SCALES:
            if cs == ics:
                continue
            pair_key = frozenset({cs, ics})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            for num in images:
                a = lookup.get((num, cs, ics))
                b = lookup.get((num, ics, cs))
                if a is None or b is None:
                    continue
                s = a + b
                sums.append(s)
                if s > 1:
                    wins += 1
                elif s < 1:
                    losses += 1
                else:
                    ties += 1
    n_decisive = wins + losses
    if n_decisive == 0 or not sums:
        return None
    p = sign_test_p(n_decisive, wins)
    return {
        "mean_sum_pct": 100 * sum(sums) / len(sums),
        "n_wins_excess_correctness": wins,
        "n_losses_deficit": losses,
        "n_ties": ties,
        "p_value": p,
    }


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
    diagonal_rows = []

    for model, outdir in MODELS.items():
        for condition in CONDITIONS:
            path = outdir / f"e1_results_{condition}_paired.json"
            if not path.exists():
                print(f"{model} / {condition}: missing {path.name}, skipping")
                continue
            lookup, images = load_grid(path)

            corner = opposite_corner_test(lookup, images)
            if corner:
                row = {"model": model, "signal_type": condition, **corner}
                corner_rows.append(row)
                print(f"[opposite-corner] {model:26s} {condition:18s} "
                      f"mean_sum={corner['mean_sum_pct']:6.2f}%  "
                      f"wins={corner['n_wins_excess_correctness']:5d} "
                      f"losses={corner['n_losses_deficit']:5d} "
                      f"ties={corner['n_ties']:5d}  p={corner['p_value']:.3e}")

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

    with open(OUT_DIR / "simple_plot_opposite_corner_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(corner_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corner_rows)

    with open(OUT_DIR / "simple_plot_diagonal_above_chance_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagonal_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagonal_rows)

    print(f"\nSaved to {OUT_DIR}/simple_plot_opposite_corner_test.csv "
          f"and {OUT_DIR}/simple_plot_diagonal_above_chance_test.csv")


if __name__ == "__main__":
    main()
