#!/usr/bin/env python3
"""
Rebuild the HTML for the main study's baseline (zero-engagement) stimuli.

Not needed to reproduce the study. The 200 baseline PNGs -- 100 correct + 100
incorrect posts by the unverified profile "Remy Ashford", every reaction, comment
and share count at zero -- are in the repository, tracked as two zip archives
under spotify_pie_plot/pie_plot_posts/baselines/{correct,incorrect}/PNGs/. Unpack
them with reproducibility/prepare_stimuli.py; only the loose PNGs are gitignored,
never the images themselves.

This script exists for the HTML, which was not archived, and for extending the
set to a new profile or claim. Run it if you want to re-derive the stimuli from
the generator rather than trust the archive, or if you are building a variant.

Settings below are not guesses. They are recovered from the archived stimuli: all
200 archived baselines are byte-identical, above the engagement bar, to the
final-generation metrics stimuli still in the repository, and the ten that also sit
loose in the tree match the archive byte for byte over the whole image. Those fix
the profile, the claim text, the chart pool and the renderer.

Run this anywhere (it only writes HTML), then render on the machine that has
Chromium and verify before trusting the output:

    python3 utils/render_html_to_png.py main-baselines
    python3 reproducibility/verify_stimuli.py --survivors

    python3 utils/generate_baseline_stimuli.py            # all 200
    python3 utils/generate_baseline_stimuli.py --only 001,003,006,022,028
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "utils"))

from post_generator_all_visible_emojis import generate_facebook_post  # noqa: E402

# The main study's chart pool. 100_pie_charts/ is a tracked, byte-identical copy of
# pie_visualizations/pop_23_5_latin_11/, which is gitignored; prefer whichever exists.
CHART_POOLS = [
    REPO_ROOT / "spotify_pie_plot/100_pie_charts",
    REPO_ROOT / "spotify_pie_plot/pie_visualizations/pop_23_5_latin_11",
]
OUT_ROOT = REPO_ROOT / "spotify_pie_plot/pie_plot_posts/baselines"

PROFILE_NAME = "Remy Ashford"
VERIFIED = False                      # the "Dr." variant is the authority condition, not this one
POST_TIME = "Today at 2:43 PM"
PROFILE_IMAGE_PATH = None

VARIANTS = {
    "correct":   ("c", "The Spotify 2026 music genres distribution has just been released. "
                       "Looks like Pop was more popular than Latin this year!"),
    "incorrect": ("i", "The Spotify 2026 music genres distribution has just been released. "
                       "Looks like Latin was more popular than Pop this year!"),
}

# Baseline means every counter visibly zero, which is what distinguishes this
# generator from the earlier "no visible 0s" one.
ZERO_REACTIONS = {k: 0 for k in ("like", "love", "haha", "wow", "sad", "angry")}


def chart_pool() -> Path:
    for p in CHART_POOLS:
        if p.is_dir() and any(p.glob("spotify_genre_pie_chart_*.png")):
            return p
    raise SystemExit(
        "No chart pool found. Expected one of:\n  " + "\n  ".join(str(p) for p in CHART_POOLS)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated image numbers, e.g. 001,003,022")
    ap.add_argument("--force", action="store_true", help="overwrite existing HTML")
    args = ap.parse_args()

    pool = chart_pool()
    wanted = set(args.only.split(",")) if args.only else {f"{i:03d}" for i in range(1, 101)}

    written = skipped = 0
    for variant, (suffix, post_text) in VARIANTS.items():
        out_dir = OUT_ROOT / variant / "html"
        out_dir.mkdir(parents=True, exist_ok=True)
        for num in sorted(wanted):
            chart = pool / f"spotify_genre_pie_chart_{num}.png"
            if not chart.exists():
                print(f"[MISS] no chart for {num}: {chart}")
                continue
            out_file = out_dir / f"{num}_remy_ashford_{suffix}.html"
            if out_file.exists() and not args.force:
                skipped += 1
                continue
            generate_facebook_post(
                profile_name=PROFILE_NAME,
                post_text=post_text,
                post_time=POST_TIME,
                reactions=dict(ZERO_REACTIONS),
                comment_count=0,
                share_count=0,
                post_image_path=str(chart),
                output_file=str(out_file),
                verified=VERIFIED,
                profile_image_path=PROFILE_IMAGE_PATH,
            )
            written += 1

    print(f"\nchart pool: {pool}")
    print(f"output:     {OUT_ROOT}/{{correct,incorrect}}/html/")
    print(f"written: {written}   skipped (already present): {skipped}")
    print("\nNext: render to PNGs/ alongside html/, then run")
    print("  python3 reproducibility/verify_stimuli.py --baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
