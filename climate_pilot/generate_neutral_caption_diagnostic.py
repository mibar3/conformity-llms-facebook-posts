"""
One-off diagnostic: is gemma4-12b's flat 100% 'scroll' response driven by the climate/energy
*framing* of the caption, rather than the chart itself? The labeled-chart fix
(generate_climate_stimuli.py) already ruled out chart legibility -- P(like) stayed saturated
even with every point's value printed on the chart. This script isolates the caption as the
remaining variable: same 25 posts, same already-labeled chart PNGs (climate_pilot/charts/), only
the surrounding post text changes -- no "2026 Energy Transition Report", no "investment" framing,
just a neutral "a new report says X was higher than Y" sentence.

Not part of the actual generalization pilot -- output goes to a separate diagnostic_neutral_caption/
folder so it can't be confused with or clobber the real climate_pilot/posts/ stimulus set.

Baseline (0 engagement) only, since that's what showed the flat response.

Run this locally (matplotlib only, no browser needed -- reuses the existing chart PNGs, doesn't
regenerate them) to produce:
  climate_pilot/diagnostic_neutral_caption/posts/{correct,incorrect}/html/{num:03d}_remy_ashford_{c,i}.html

HTML -> PNG rendering is a separate, later step (needs Selenium + Chromium) -- see
render_neutral_caption_diagnostic.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "utils"))
sys.path.insert(0, str(ROOT_DIR))
from post_generator_all_visible_emojis import generate_facebook_post  # noqa: E402
from climate_pilot.generate_climate_stimuli import (  # noqa: E402
    generate_solar_wind_series, N_POSTS, PROFILE_NAME, POST_TIME, VERIFIED, PROFILE_IMAGE_PATH,
)

OUT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = OUT_DIR / "charts"
DIAG_DIR = OUT_DIR / "diagnostic_neutral_caption"
POSTS_DIR = DIAG_DIR / "posts"


def make_neutral_claim_text(solar_wins: bool, correct: bool) -> str:
    # Same sentence structure and same solar/wind ground truth as the climate version (so it's a
    # like-for-like caption swap), but no report name, no "energy"/"investment"/"climate" framing
    # -- just neutral reporting language.
    claimed_winner_is_solar = solar_wins if correct else (not solar_wins)
    winner_name = "solar" if claimed_winner_is_solar else "wind"
    loser_name = "wind" if claimed_winner_is_solar else "solar"
    return f"A new report has just been released. Looks like {winner_name} ended up higher than {loser_name} by Year 10!"


def main():
    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    for num in range(1, N_POSTS + 1):
        _, _, solar_wins = generate_solar_wind_series(num)
        chart_path = CHARTS_DIR / f"{num:03d}.png"

        for variant, correct in (("correct", True), ("incorrect", False)):
            suffix = "c" if correct else "i"
            post_text = make_neutral_claim_text(solar_wins, correct)
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
