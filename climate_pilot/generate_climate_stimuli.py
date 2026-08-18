"""
Generalization pilot: solar-vs-wind line chart, climate-framed claim (supervisor item 7 /
Chapter 7 Limitation 2 / Chapter 8 item 4 -- content-domain and chart-type generalization).

Design, matching the reasoning already established for the main study's pie chart:
- Chart type: line chart. Per Pandey & Ottley (2025) (THESIS.md Section 2.9), line charts are
  in the same "VLMs already read this well" tier as pie charts -- chosen specifically to avoid
  reopening the perception confound the pie chart was originally picked to avoid.
- Claim: a single-point magnitude comparison ("solar share was higher than wind share by Year
  6"), not a rate-of-change claim -- kept at the same reading difficulty as "Pop was more
  popular than Latin" on purpose, so a null/positive result here isn't confounded by the new
  claim being a harder reading task than the original.
- Data: entirely fabricated, generic "Year 1".."Year 6" x-axis (not real calendar years) -- so
  no model's training-data cutoff can matter, and no model can lean on a real-world prior about
  actual solar/wind adoption trends. Same principle as the original Pop/Latin percentages,
  which were never real Spotify data either.
- Same "correct"/"incorrect" pairing trick as the original: one chart per post number, shared
  identically between both text variants -- only the caption's claimed winner changes.

Run this locally (matplotlib only, no browser needed) to produce:
  climate_pilot/charts/{num:03d}.png                      -- one chart per post, shared by both variants
  climate_pilot/posts/{correct,incorrect}/html/{num:03d}_remy_ashford_{c,i}.html            (baseline, 0 engagement)
  climate_pilot/posts/{correct,incorrect}/html/metrics/realistic/{scale}/{num:03d}_remy_ashford_{c,i}.html
  climate_pilot/ground_truth_climate.csv

HTML -> PNG rendering is a separate, later step (needs Selenium + a real Chromium binary, not
available in this environment) -- see render_climate_html_to_png.py.
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
SCALE_VALUES = [10, 100, 1000, 10000, 100000, 1000000]  # matches REACTION_VALUES elsewhere
YEAR_LABELS = [f"Year {i}" for i in range(1, 7)]  # 6 points, generic (not real calendar years)
REACTION_TYPES = ["like", "love", "haha", "wow", "sad", "angry"]

PROFILE_NAME = "Remy Ashford"
POST_TIME = "Today at 2:43 PM"
VERIFIED = False
PROFILE_IMAGE_PATH = None


def generate_solar_wind_series(post_num: int):
    """Fabricated 6-point series for solar/wind capacity share (%). Deterministic per post
    number so re-running this script reproduces the identical dataset. Returns
    (solar_values, wind_values, solar_wins: bool)."""
    rng = random.Random(SEED + post_num)

    solar_start = rng.uniform(15, 30)
    wind_start = rng.uniform(15, 30)
    solar_drift = rng.uniform(-3, 6)  # per-year average change
    wind_drift = rng.uniform(-3, 6)

    solar_values, wind_values = [], []
    s, w = solar_start, wind_start
    for _ in YEAR_LABELS:
        s = max(2.0, s + solar_drift + rng.uniform(-1.5, 1.5))
        w = max(2.0, w + wind_drift + rng.uniform(-1.5, 1.5))
        solar_values.append(round(s, 1))
        wind_values.append(round(w, 1))

    # Force a clean, unambiguous winner at the final point (avoid near-tie endpoints, which
    # would make the "correct" answer genuinely ambiguous from the chart rather than a clean
    # ground truth) -- mirrors how the pie chart's fixed Pop/Latin values were chosen to be
    # clearly separated, not a coin-flip-close pair.
    solar_wins = rng.random() < 0.5
    min_gap = 4.0
    if solar_wins and solar_values[-1] <= wind_values[-1] + min_gap:
        solar_values[-1] = wind_values[-1] + min_gap + rng.uniform(0, 3)
    elif not solar_wins and wind_values[-1] <= solar_values[-1] + min_gap:
        wind_values[-1] = solar_values[-1] + min_gap + rng.uniform(0, 3)

    return solar_values, wind_values, solar_wins


def make_line_chart(solar_values, wind_values, output_path: Path):
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(YEAR_LABELS, solar_values, label="Solar", color="#f39c12",
             linewidth=2.5, marker="o", markersize=6)
    ax.plot(YEAR_LABELS, wind_values, label="Wind", color="#3498db",
             linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel("Share of total generation capacity (%)")
    ax.set_title("Renewable Energy Capacity Share")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_claim_text(solar_wins: bool, correct: bool) -> str:
    # correct=True -> caption states the TRUE winner; correct=False -> states the opposite.
    claimed_winner_is_solar = solar_wins if correct else (not solar_wins)
    winner_name = "solar" if claimed_winner_is_solar else "wind"
    loser_name = "wind" if claimed_winner_is_solar else "solar"
    return (
        "The 2026 Global Energy Transition Report has just been released. "
        f"Looks like {winner_name} capacity share ended up higher than {loser_name} "
        f"capacity share by Year 6!"
    )


def make_log_reactions(scale_value: int, jitter: float = 0.10, seed: int = None) -> dict:
    """Same 'realistic' log-weighted reaction-type allocation used for the main study's metrics
    condition (ported from utils/updated_post_generator_all_visible_emojis.ipynb) -- likes get
    the largest share, angry the smallest, with per-reaction jitter."""
    rng = random.Random(seed)
    n = len(REACTION_TYPES)
    log_weights = [math.log(n + 1 - i) for i in range(n)]
    total_weight = sum(log_weights)
    shares = [w / total_weight for w in log_weights]
    rng.shuffle(shares)
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
        solar_values, wind_values, solar_wins = generate_solar_wind_series(num)
        chart_path = CHARTS_DIR / f"{num:03d}.png"
        make_line_chart(solar_values, wind_values, chart_path)

        for variant, correct in (("correct", True), ("incorrect", False)):
            suffix = "c" if correct else "i"
            post_text = make_claim_text(solar_wins, correct)

            # Baseline: 0 engagement, no metrics shown.
            baseline_out = POSTS_DIR / variant / "html" / f"{num:03d}_remy_ashford_{suffix}.html"
            generate_facebook_post(
                profile_name=PROFILE_NAME, post_text=post_text, post_time=POST_TIME,
                reactions={}, comment_count=0, share_count=0,
                profile_image_path=PROFILE_IMAGE_PATH, post_image_path=str(chart_path),
                output_file=str(baseline_out), verified=VERIFIED,
            )

            # 6 engagement scales, "realistic" (metrics) reaction distribution -- the condition
            # that showed the strongest conformity effect in the main study (Section 6.2).
            for scale in SCALE_VALUES:
                reactions = make_log_reactions(scale, seed=SEED + num + scale)
                scaled_out = (POSTS_DIR / variant / "html" / "metrics" / "realistic" / str(scale)
                              / f"{num:03d}_remy_ashford_{suffix}.html")
                generate_facebook_post(
                    profile_name=PROFILE_NAME, post_text=post_text, post_time=POST_TIME,
                    reactions=reactions, comment_count=scale, share_count=scale,
                    profile_image_path=PROFILE_IMAGE_PATH, post_image_path=str(chart_path),
                    output_file=str(scaled_out), verified=VERIFIED,
                )

        ground_truth_rows.append({
            "num": f"{num:03d}",
            "solar_year6": solar_values[-1], "wind_year6": wind_values[-1],
            "solar_series": ";".join(str(v) for v in solar_values),
            "wind_series": ";".join(str(v) for v in wind_values),
            "solar_wins": solar_wins,
            "post_claim_correct_text": make_claim_text(solar_wins, True),
            "post_claim_incorrect_text": make_claim_text(solar_wins, False),
        })

    gt_path = OUT_DIR / "ground_truth_climate.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ground_truth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ground_truth_rows)

    print(f"Done. {N_POSTS} posts x 2 variants x (1 baseline + {len(SCALE_VALUES)} scales) "
          f"= {N_POSTS * 2 * (1 + len(SCALE_VALUES))} HTML files written under {POSTS_DIR}")
    print(f"Ground truth: {gt_path}")


if __name__ == "__main__":
    main()
