"""
Second chart-type diagnostic for gemma4-12b's flat baseline response. Value-labeling the line
chart didn't fix it (generate_climate_stimuli.py) and neither did a neutral, non-climate caption
(generate_neutral_caption_diagnostic.py) -- both content variables are ruled out. This isolates
the remaining variable: chart *type/visual genre*. Pandey & Ottley (2025) (THESIS.md Section 2.9)
found pie charts, line charts, scatterplots, and stacked area charts in the same "VLMs already
read this well" tier -- scatterplot is the most literature-faithful next chart type to try,
since it stays inside that tier rather than introducing a new confound the way an unstudied
chart type (e.g. bar) would.

Same 25 posts, same fabricated solar/wind data, same value-per-point labeling, same climate
caption (generate_climate_stimuli.make_claim_text) -- only the chart itself is now a scatterplot
instead of a line chart with markers (no connecting lines).

Not part of the actual generalization pilot -- output goes to a separate diagnostic_scatter_chart/
folder so it can't be confused with or clobber the real climate_pilot/posts/ stimulus set.

Baseline (0 engagement) only, since that's what showed the flat response.

Run this locally (matplotlib only, no browser needed) to produce:
  climate_pilot/diagnostic_scatter_chart/charts/{num:03d}.png
  climate_pilot/diagnostic_scatter_chart/posts/{correct,incorrect}/html/{num:03d}_remy_ashford_{c,i}.html

HTML -> PNG rendering is a separate, later step (needs Selenium + Chromium) -- see
render_scatter_diagnostic.py.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "utils"))
sys.path.insert(0, str(ROOT_DIR))
from post_generator_all_visible_emojis import generate_facebook_post  # noqa: E402
from climate_pilot.generate_climate_stimuli import (  # noqa: E402
    generate_solar_wind_series, make_claim_text, YEAR_LABELS, N_POSTS,
    PROFILE_NAME, POST_TIME, VERIFIED, PROFILE_IMAGE_PATH,
)

OUT_DIR = Path(__file__).resolve().parent
DIAG_DIR = OUT_DIR / "diagnostic_scatter_chart"
CHARTS_DIR = DIAG_DIR / "charts"
POSTS_DIR = DIAG_DIR / "posts"


def make_scatter_chart(solar_values, wind_values, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(YEAR_LABELS))
    ax.scatter(x, solar_values, label="Solar", color="#f39c12", s=90, zorder=3)
    ax.scatter(x, wind_values, label="Wind", color="#3498db", s=90, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(YEAR_LABELS)
    ax.set_ylabel("Annual investment ($ billions)")
    ax.set_title("Renewable Energy Investment")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    for i, (s, w) in enumerate(zip(solar_values, wind_values)):
        solar_off, wind_off = ((0, 10), (0, -16)) if s >= w else ((0, -16), (0, 10))
        ax.annotate(f"{s:.1f}", (i, s), textcoords="offset points", xytext=solar_off,
                    fontsize=7.5, color="#c8790a", ha="center")
        ax.annotate(f"{w:.1f}", (i, w), textcoords="offset points", xytext=wind_off,
                    fontsize=7.5, color="#2a72a8", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    for num in range(1, N_POSTS + 1):
        solar_values, wind_values, solar_wins = generate_solar_wind_series(num)
        chart_path = CHARTS_DIR / f"{num:03d}.png"
        make_scatter_chart(solar_values, wind_values, chart_path)

        for variant, correct in (("correct", True), ("incorrect", False)):
            suffix = "c" if correct else "i"
            post_text = make_claim_text(solar_wins, correct)
            out_html = POSTS_DIR / variant / "html" / f"{num:03d}_remy_ashford_{suffix}.html"
            generate_facebook_post(
                profile_name=PROFILE_NAME, post_text=post_text, post_time=POST_TIME,
                reactions={}, comment_count=0, share_count=0,
                profile_image_path=PROFILE_IMAGE_PATH, post_image_path=str(chart_path),
                output_file=str(out_html), verified=VERIFIED,
            )

    print(f"\nDone. {N_POSTS} posts x 2 variants = {N_POSTS * 2} HTML files written under {POSTS_DIR}")


if __name__ == "__main__":
    main()
