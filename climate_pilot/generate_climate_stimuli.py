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
- Data: entirely fabricated, generic "Year 1".."Year 10" x-axis (not real calendar years) -- so
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
YEAR_LABELS = [f"Year {i}" for i in range(1, 11)]  # 10 points, generic (not real calendar years)
REACTION_TYPES = ["like", "love", "haha", "wow", "sad", "angry"]

PROFILE_NAME = "Remy Ashford"
POST_TIME = "Today at 2:43 PM"
VERIFIED = False
PROFILE_IMAGE_PATH = None


def generate_solar_wind_series(post_num: int):
    """Fabricated series for solar/wind annual investment ($ billions) -- chosen over a
    capacity-share (%) framing since "how much money was invested" needs no domain knowledge of
    grid-capacity accounting to read, keeping the claim at the same intuitive difficulty as the
    original Pop-vs-Latin percentage comparison. Deterministic per post number so re-running
    this script reproduces the identical dataset. Returns (solar_values, wind_values, solar_wins: bool).

    Solar is the fixed true winner in every single post -- matches the original study's
    convention exactly (Pop was the fixed true winner over Latin in every one of the 50 pie
    charts, e.g. fixed_pop=23.5/fixed_latin=11.0 in one batch, fixed_pop=55/fixed_latin=11 in
    another -- the magnitude of the win varies, the *direction* never does).

    Solar is generated as wind's own trajectory plus a gap that moves smoothly from a starting
    value toward a comfortably positive final value, rather than generating solar independently
    and patching only the last point to force a win -- the earlier version could (and did, e.g.
    post 001) produce a solar line trending down for years and then jumping sharply at the very
    last point, which looks unrealistic. This way the "solar pulls ahead" story is spread across
    the whole series."""
    rng = random.Random(SEED + post_num)
    n = len(YEAR_LABELS)

    wind_start = rng.uniform(15, 30)
    wind_drift = rng.uniform(-3, 6)  # per-year average change
    wind_values = []
    w = wind_start
    for _ in YEAR_LABELS:
        w = max(2.0, w + wind_drift + rng.uniform(-1.5, 1.5))
        wind_values.append(w)

    min_gap = 4.0
    gap_start = rng.uniform(-6, 2)          # solar can start behind or roughly tied
    gap_end = min_gap + rng.uniform(0, 4)    # but always ends comfortably ahead
    solar_values = []
    for i in range(n):
        t = i / (n - 1)
        target_gap = gap_start + (gap_end - gap_start) * t  # smooth linear taper, not a jump
        noisy_gap = target_gap + rng.uniform(-1.2, 1.2)
        solar_values.append(max(2.0, wind_values[i] + noisy_gap))

    solar_values = [round(v, 1) for v in solar_values]
    wind_values = [round(v, 1) for v in wind_values]

    # Safety net only -- rounding/clamping noise could in principle shave the final gap below
    # the minimum; should rarely trigger given the construction above (gap_end's floor already
    # sits at min_gap, and per-point noise is small relative to it), so when it does trigger the
    # correction needed is small too -- not a dramatic peak. A flat or gently declining tail is
    # fine; the only thing this guards against is the *ratio* ground truth (solar must end up
    # ahead by min_gap), not the shape of the last few points.
    solar_wins = True
    if solar_values[-1] <= wind_values[-1] + min_gap:
        solar_values[-1] = round(wind_values[-1] + min_gap + rng.uniform(0, 2), 1)

    return solar_values, wind_values, solar_wins


def make_line_chart(solar_values, wind_values, output_path: Path):
    # Deliberately always overwrites (no skip-if-exists) -- chart generation is cheap (pure
    # matplotlib, no browser), and a stale skip here previously caused charts to silently keep
    # showing an old dataset after the generation logic changed, while the HTML/captions moved
    # on to the new one -- a real mismatch this caused once already.
    #
    # All points labeled with their value, not just Year 10 -- the unlabeled version left models
    # needing to visually estimate line height with no printed number to read, unlike the main
    # study's pie chart where the compared quantities are printed on the slices. Three of four
    # models showed zero content-conditioned behavior on the unlabeled chart (see e1_climate
    # baseline/metrics results). Starting with every point labeled (not just Year 10) since the
    # main study already found that simplifying a chart (e1_simple_plot/_bigfont) didn't move
    # perception -- more information first, fall back to Year-10-only if this doesn't help either.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(YEAR_LABELS, solar_values, label="Solar", color="#f39c12",
             linewidth=2.5, marker="o", markersize=6)
    ax.plot(YEAR_LABELS, wind_values, label="Wind", color="#3498db",
             linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel("Annual investment ($ billions)")
    ax.set_title("Renewable Energy Investment")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    for i, (s, w) in enumerate(zip(solar_values, wind_values)):
        solar_off, wind_off = ((0, 10), (0, -14)) if s >= w else ((0, -14), (0, 10))
        ax.annotate(f"{s:.1f}", (i, s), textcoords="offset points", xytext=solar_off,
                    fontsize=7.5, color="#c8790a", ha="center")
        ax.annotate(f"{w:.1f}", (i, w), textcoords="offset points", xytext=wind_off,
                    fontsize=7.5, color="#2a72a8", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_claim_text(solar_wins: bool, correct: bool) -> str:
    # correct=True -> caption states the TRUE winner; correct=False -> states the opposite.
    claimed_winner_is_solar = solar_wins if correct else (not solar_wins)
    winner_name = "solar" if claimed_winner_is_solar else "wind"
    loser_name = "wind" if claimed_winner_is_solar else "solar"
    return (
        "The 2026 Energy Transition Report has just been released. "
        f"Looks like {winner_name} investment ended up higher than {loser_name} "
        f"investment by Year 10!"
    )


def make_log_reactions(scale_value: int, jitter: float = 0.10, seed: int = None) -> dict:
    """Same 'realistic' log-weighted reaction-type allocation used for the main study's metrics
    condition (ported from the production cell of utils/updated_post_generator_all_visible_emojis.ipynb,
    not the earlier draft cell that shuffled shares -- the shares are deliberately left in fixed
    order so 'like' always gets the largest share and 'angry' always the smallest, matching the
    main study's documented realistic-condition behavior exactly, jitter aside)."""
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
        solar_values, wind_values, solar_wins = generate_solar_wind_series(num)
        chart_path = CHARTS_DIR / f"{num:03d}.png"
        make_line_chart(solar_values, wind_values, chart_path)

        for variant, correct in (("correct", True), ("incorrect", False)):
            suffix = "c" if correct else "i"
            post_text = make_claim_text(solar_wins, correct)

            # Baseline: 0 engagement, shown as explicit zeros (not an empty reactions dict --
            # that hides the whole reaction row instead of rendering "0 0 0 0 0 0", the same
            # bug already documented and fixed once for the bigger-font pilot's baseline
            # stimuli -- see utils/post_generator_all_visible_emojis.py's build_reaction_bubbles_html).
            baseline_out = POSTS_DIR / variant / "html" / f"{num:03d}_remy_ashford_{suffix}.html"
            generate_facebook_post(
                profile_name=PROFILE_NAME, post_text=post_text, post_time=POST_TIME,
                reactions={"like": 0, "love": 0, "haha": 0, "wow": 0, "sad": 0, "angry": 0},
                comment_count=0, share_count=0,
                profile_image_path=PROFILE_IMAGE_PATH, post_image_path=str(chart_path),
                output_file=str(baseline_out), verified=VERIFIED,
            )

            # 6 engagement scales, "realistic" (metrics) reaction distribution -- the condition
            # that showed the strongest conformity effect in the main study (Section 6.2).
            # seed=num (not scale-dependent) and the 0.08/0.04 comment/share fractions match the
            # main study's production generator exactly, not just the same shape.
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
            "solar_final_year": solar_values[-1], "wind_final_year": wind_values[-1],
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
