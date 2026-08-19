"""
Mini chart-type benchmark for gemma4-12b's flat baseline response (100%/0% like rate, no
correct/incorrect separation, unmoved by chart labels, caption wording, or a scatterplot -- see
the three diagnostics in climate_pilot/: generate_climate_stimuli.py's labeling, and the
generate_neutral_caption_diagnostic.py / generate_scatter_diagnostic.py folders).

Rather than testing one more chart type at a time through the full generate -> Chromium-render ->
GPU-notebook round trip, this generates several chart types in a single batch so they can be
tested together in one inference pass (run_benchmark_gemma12b.py) against just one model. Only
promote whichever chart type (if any) breaks the flat response into a real diagnostic/pilot.

10 posts (001-010, not the full 25) -- this is a fast benchmark pass, not a replication. Same
fabricated solar/wind data and climate caption as the real pilot (both already ruled out as the
cause), baseline only (0 engagement, the condition that showed the flat response).

Chart types:
  bar_grouped     -- grouped bar chart, all 10 years, two bars per year, value-labeled.
  bar_single_year -- minimal two-bar comparison at Year 10 only, closest analog to the main
                     study's pie chart (one glance, one pair of numbers).
  area_overlay    -- two semi-transparent overlaid area fills (not stacked -- stacking would sum
                     the series and change what "higher" means). Area charts are in the same
                     VLM-legible tier as pie/line/scatter per Pandey & Ottley (2025).

Run this locally (matplotlib only, no browser needed) to produce:
  climate_pilot/chart_type_benchmark/<chart_type>/charts/{num:03d}.png
  climate_pilot/chart_type_benchmark/<chart_type>/posts/{correct,incorrect}/html/{num:03d}_remy_ashford_{c,i}.html

HTML -> PNG rendering is a separate, later step (needs Selenium + Chromium) -- see
render_benchmark_charts.py.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "utils"))
sys.path.insert(0, str(ROOT_DIR))
from post_generator_all_visible_emojis import generate_facebook_post  # noqa: E402
from climate_pilot.generate_climate_stimuli import (  # noqa: E402
    generate_solar_wind_series, make_claim_text, YEAR_LABELS,
    PROFILE_NAME, POST_TIME, VERIFIED, PROFILE_IMAGE_PATH,
)

N_POSTS = 10  # 001-010 only -- fast benchmark pass
BENCH_DIR = Path(__file__).resolve().parent
SOLAR_COLOR, WIND_COLOR = "#f39c12", "#3498db"


def make_bar_grouped_chart(solar_values, wind_values, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(YEAR_LABELS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, solar_values, width, label="Solar", color=SOLAR_COLOR)
    ax.bar(x + width / 2, wind_values, width, label="Wind", color=WIND_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(YEAR_LABELS)
    ax.set_ylabel("Annual investment ($ billions)")
    ax.set_title("Renewable Energy Investment")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    for i, (s, w) in enumerate(zip(solar_values, wind_values)):
        ax.annotate(f"{s:.0f}", (i - width / 2, s), textcoords="offset points", xytext=(0, 4),
                    fontsize=7, color=SOLAR_COLOR, ha="center")
        ax.annotate(f"{w:.0f}", (i + width / 2, w), textcoords="offset points", xytext=(0, 4),
                    fontsize=7, color=WIND_COLOR, ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_bar_single_year_chart(solar_values, wind_values, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    solar_last, wind_last = solar_values[-1], wind_values[-1]

    fig, ax = plt.subplots(figsize=(5, 5.5))
    bars = ax.bar(["Solar", "Wind"], [solar_last, wind_last], color=[SOLAR_COLOR, WIND_COLOR], width=0.5)
    ax.set_ylabel("Year 10 investment ($ billions)")
    ax.set_title("Renewable Energy Investment -- Year 10")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    for bar, val in zip(bars, [solar_last, wind_last]):
        ax.annotate(f"${val:.1f}B", (bar.get_x() + bar.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 6), fontsize=13, fontweight="bold",
                    ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_area_overlay_chart(solar_values, wind_values, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(YEAR_LABELS))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(x, solar_values, color=SOLAR_COLOR, alpha=0.45, label="Solar")
    ax.plot(x, solar_values, color=SOLAR_COLOR, linewidth=2)
    ax.fill_between(x, wind_values, color=WIND_COLOR, alpha=0.45, label="Wind")
    ax.plot(x, wind_values, color=WIND_COLOR, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(YEAR_LABELS)
    ax.set_ylabel("Annual investment ($ billions)")
    ax.set_title("Renewable Energy Investment")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    for i, (s, w) in enumerate(zip(solar_values, wind_values)):
        solar_off, wind_off = ((0, 8), (0, -14)) if s >= w else ((0, -14), (0, 8))
        ax.annotate(f"{s:.1f}", (i, s), textcoords="offset points", xytext=solar_off,
                    fontsize=7.5, color="#8a5209", ha="center")
        ax.annotate(f"{w:.1f}", (i, w), textcoords="offset points", xytext=wind_off,
                    fontsize=7.5, color="#1c5279", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


CHART_MAKERS = {
    "bar_grouped": make_bar_grouped_chart,
    "bar_single_year": make_bar_single_year_chart,
    "area_overlay": make_area_overlay_chart,
}


def main():
    for chart_type, make_chart in CHART_MAKERS.items():
        type_dir = BENCH_DIR / chart_type
        charts_dir = type_dir / "charts"
        posts_dir = type_dir / "posts"

        for num in range(1, N_POSTS + 1):
            solar_values, wind_values, solar_wins = generate_solar_wind_series(num)
            chart_path = charts_dir / f"{num:03d}.png"
            make_chart(solar_values, wind_values, chart_path)

            for variant, correct in (("correct", True), ("incorrect", False)):
                suffix = "c" if correct else "i"
                post_text = make_claim_text(solar_wins, correct)
                out_html = posts_dir / variant / "html" / f"{num:03d}_remy_ashford_{suffix}.html"
                generate_facebook_post(
                    profile_name=PROFILE_NAME, post_text=post_text, post_time=POST_TIME,
                    reactions={}, comment_count=0, share_count=0,
                    profile_image_path=PROFILE_IMAGE_PATH, post_image_path=str(chart_path),
                    output_file=str(out_html), verified=VERIFIED,
                )
        print(f"[{chart_type}] {N_POSTS} posts x 2 variants written under {posts_dir}")

    print(f"\nDone. {len(CHART_MAKERS)} chart types x {N_POSTS} posts x 2 variants = "
          f"{len(CHART_MAKERS) * N_POSTS * 2} HTML files.")


if __name__ == "__main__":
    main()
