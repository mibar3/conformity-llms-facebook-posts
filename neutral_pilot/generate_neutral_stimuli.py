"""
Follow-up to the climate generalization pilot (climate_pilot/generate_climate_stimuli.py):
same line chart, same fabricated data-generation logic, same engagement manipulation --
only the topic is swapped from a climate/renewable-energy frame to a maximally generic,
content-free one ("Product A" vs. "Product B" quarterly revenue). Purpose: isolate whether
Gemma-12B's climate-pilot inversion (it prefers the LOWER-engagement post in all 21 of 21
tested off-diagonal scale-pairs, the opposite of every other model's normal conformity
direction on the same pilot, and the opposite of Gemma-12B's own behavior on the main
Pop-vs-Latin study) is specific to the climate/renewable-energy content, or a general
property of this model's behavior on this chart type + engagement manipulation regardless
of topic.

Design, deliberately unchanged from the climate pilot except for labels/text:
- Chart type: line chart, same colors, markers, per-point value labels, and layout.
- Data: the exact same fabricated-series generation function and RNG seeding as the climate
  pilot (SEED=42, per-post seeding via SEED + post_num) -- reusing the identical algorithm
  guarantees the two pilots' underlying numeric series are drawn from the same process, so a
  difference in model behavior between them can't be attributed to the data pattern itself.
- Claim: the same single-point magnitude comparison structure ("X ended up higher than Y by
  Year 10"), same reading difficulty as both the climate pilot and the original study.
- Topic: "Product A" / "Product B" quarterly revenue ($ billions) -- chosen over another
  real-world topic pair specifically to avoid introducing a *different* content confound
  (brand familiarity, category association); this isolates "is it climate/political framing
  specifically" from "is it any topic at all" as cleanly as possible.
- Scope: Gemma-12B only, per the supervisor's request following the climate-pilot
  triple-check (2026-08-28) -- this is a targeted follow-up, not a full 6-model rerun.

Run this locally (matplotlib only, no browser needed) to produce:
  neutral_pilot/charts/{num:03d}.png                      -- one chart per post, shared by both variants
  neutral_pilot/posts/{correct,incorrect}/html/{num:03d}_remy_ashford_{c,i}.html            (baseline, 0 engagement)
  neutral_pilot/posts/{correct,incorrect}/html/metrics/realistic/{scale}/{num:03d}_remy_ashford_{c,i}.html
  neutral_pilot/ground_truth_neutral.csv

HTML -> PNG rendering is a separate, later step (needs Selenium + a real Chromium binary) --
see climate_pilot/render_climate_html_to_png.py for the pattern to reuse (swap the input/output
directories to neutral_pilot/).
"""
import csv
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "utils"))
from post_generator_all_visible_emojis import generate_facebook_post  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = OUT_DIR / "charts"
POSTS_DIR = OUT_DIR / "posts"

N_POSTS = 25
SEED = 42
SCALE_VALUES = [10, 100, 1000, 10000, 100000, 1000000]  # matches the climate pilot exactly
YEAR_LABELS = [f"Year {i}" for i in range(1, 11)]  # 10 points, generic (not real calendar years)
REACTION_TYPES = ["like", "love", "haha", "wow", "sad", "angry"]

PROFILE_NAME = "Remy Ashford"
POST_TIME = "Today at 2:43 PM"
VERIFIED = False
PROFILE_IMAGE_PATH = None


def generate_product_series(post_num: int):
    """Identical algorithm and seeding to the climate pilot's generate_solar_wind_series --
    only the variable names changed. Deterministic per post number so re-running this script
    reproduces the identical dataset, and reproduces the same *numeric* series as the climate
    pilot's post of the same number, since the RNG draws never depend on topic labels.
    Returns (product_a_values, product_b_values, product_a_wins: bool).

    Product A is the fixed true winner in every single post, matching both the climate pilot
    and the original study's convention (the *direction* of the win never varies, only its
    magnitude).
    """
    rng = random.Random(SEED + post_num)
    n = len(YEAR_LABELS)

    b_start = rng.uniform(15, 30)
    b_drift = rng.uniform(-3, 6)  # per-year average change
    b_values = []
    b = b_start
    for _ in YEAR_LABELS:
        b = max(2.0, b + b_drift + rng.uniform(-1.5, 1.5))
        b_values.append(b)

    min_gap = 4.0
    gap_start = rng.uniform(-6, 2)          # Product A can start behind or roughly tied
    gap_end = min_gap + rng.uniform(0, 4)    # but always ends comfortably ahead
    a_values = []
    for i in range(n):
        t = i / (n - 1)
        target_gap = gap_start + (gap_end - gap_start) * t  # smooth linear taper, not a jump
        noisy_gap = target_gap + rng.uniform(-1.2, 1.2)
        a_values.append(max(2.0, b_values[i] + noisy_gap))

    a_values = [round(v, 1) for v in a_values]
    b_values = [round(v, 1) for v in b_values]

    product_a_wins = True
    if a_values[-1] <= b_values[-1] + min_gap:
        a_values[-1] = round(b_values[-1] + min_gap + rng.uniform(0, 2), 1)

    return a_values, b_values, product_a_wins


def make_line_chart(a_values, b_values, output_path: Path):
    # Same styling, colors, and per-point labeling as the climate pilot's make_line_chart --
    # only the legend/title/axis text changed.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(YEAR_LABELS, a_values, label="Product A", color="#f39c12",
             linewidth=2.5, marker="o", markersize=6)
    ax.plot(YEAR_LABELS, b_values, label="Product B", color="#3498db",
             linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel("Quarterly revenue ($ billions)")
    ax.set_title("Quarterly Product Revenue")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    for i, (a, b) in enumerate(zip(a_values, b_values)):
        a_off, b_off = ((0, 10), (0, -14)) if a >= b else ((0, -14), (0, 10))
        ax.annotate(f"{a:.1f}", (i, a), textcoords="offset points", xytext=a_off,
                    fontsize=7.5, color="#c8790a", ha="center")
        ax.annotate(f"{b:.1f}", (i, b), textcoords="offset points", xytext=b_off,
                    fontsize=7.5, color="#2a72a8", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_claim_text(product_a_wins: bool, correct: bool) -> str:
    # correct=True -> caption states the TRUE winner; correct=False -> states the opposite.
    claimed_winner_is_a = product_a_wins if correct else (not product_a_wins)
    winner_name = "Product A" if claimed_winner_is_a else "Product B"
    loser_name = "Product B" if claimed_winner_is_a else "Product A"
    return (
        "The 2026 Market Report has just been released. "
        f"Looks like {winner_name} revenue ended up higher than {loser_name} "
        f"revenue by Year 10!"
    )


def make_log_reactions(scale_value: int, jitter: float = 0.10, seed: int = None) -> dict:
    """Identical to the climate pilot's make_log_reactions -- same fixed reaction-share order
    and jitter, matching the main study's documented realistic-condition behavior."""
    rng = random.Random(seed)
    n = len(REACTION_TYPES)
    log_weights = [math.log(n + 1 - i) for i in range(n)]
    total_weight = sum(log_weights)
    shares = [w / total_weight for w in log_weights]
    reactions = {}
    for emoji, share in zip(REACTION_TYPES, shares):
        base = scale_value * share
        factor = 1 + rng.uniform(-jitter, jitter)
        reactions[emoji] = max(1, round(base * factor))
    return reactions


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth_rows = []

    for num in range(1, N_POSTS + 1):
        a_values, b_values, product_a_wins = generate_product_series(num)
        chart_path = CHARTS_DIR / f"{num:03d}.png"
        make_line_chart(a_values, b_values, chart_path)

        for variant, correct in (("correct", True), ("incorrect", False)):
            suffix = "c" if correct else "i"
            post_text = make_claim_text(product_a_wins, correct)

            # Baseline: 0 engagement, shown as explicit zeros -- same fix as the climate pilot.
            baseline_out = POSTS_DIR / variant / "html" / f"{num:03d}_remy_ashford_{suffix}.html"
            generate_facebook_post(
                profile_name=PROFILE_NAME, post_text=post_text, post_time=POST_TIME,
                reactions={"like": 0, "love": 0, "haha": 0, "wow": 0, "sad": 0, "angry": 0},
                comment_count=0, share_count=0,
                profile_image_path=PROFILE_IMAGE_PATH, post_image_path=str(chart_path),
                output_file=str(baseline_out), verified=VERIFIED,
            )

            # 6 engagement scales, "realistic" (metrics) reaction distribution -- same
            # seed=num (not scale-dependent) and comment/share fractions as the climate pilot.
            for scale in SCALE_VALUES:
                reactions = make_log_reactions(scale, seed=num)
                comment_count = max(1, round(scale * 0.08))
                share_count = max(1, round(scale * 0.04))
                scaled_out = (POSTS_DIR / variant / "html" / "metrics" / "realistic" / str(scale)
                              / f"{num:03d}_remy_ashford_{suffix}.html")
                generate_facebook_post(
                    profile_name=PROFILE_NAME, post_text=post_text, post_time=POST_TIME,
                    reactions=reactions, comment_count=comment_count, share_count=share_count,
                    profile_image_path=PROFILE_IMAGE_PATH, post_image_path=str(chart_path),
                    output_file=str(scaled_out), verified=VERIFIED,
                )

        ground_truth_rows.append({
            "num": f"{num:03d}",
            "product_a_final_year": a_values[-1], "product_b_final_year": b_values[-1],
            "product_a_series": ";".join(str(v) for v in a_values),
            "product_b_series": ";".join(str(v) for v in b_values),
            "product_a_wins": product_a_wins,
            "post_claim_correct_text": make_claim_text(product_a_wins, True),
            "post_claim_incorrect_text": make_claim_text(product_a_wins, False),
        })

    gt_path = OUT_DIR / "ground_truth_neutral.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ground_truth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ground_truth_rows)

    print(f"Done. {N_POSTS} posts x 2 variants x (1 baseline + {len(SCALE_VALUES)} scales) "
          f"= {N_POSTS * 2 * (1 + len(SCALE_VALUES))} HTML files written under {POSTS_DIR}")
    print(f"Ground truth: {gt_path}")


if __name__ == "__main__":
    main()
