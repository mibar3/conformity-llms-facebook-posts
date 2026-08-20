"""
Authority-condition stimulus generation: "Dr. Remy Ashford" + verified badge.

Produces a complete parallel copy of the main study's Remy Ashford stimulus set, identical in
every respect except the poster's identity — same 100 original all-genres pie charts, same
claim texts, same engagement scales, same reaction values. Only `PROFILE_NAME` and `VERIFIED`
differ from the main study's generator (the production cells of
`utils/updated_post_generator_all_visible_emojis.ipynb`).

Motivation: test whether an authority cue in the poster's identity shifts either competence or
conformity. See docs/SUPERVISOR_MEETING_NOTES.md.

--------------------------------------------------------------------------------------------
FILENAMES ARE DELIBERATELY UNCHANGED — DO NOT "FIX" THEM
--------------------------------------------------------------------------------------------
Files are still named `{NNN}_remy_ashford_{c,i}.html`, NOT `{NNN}_dr_remy_ashford_{c,i}.html`.
`experiments/e1/e1_utils/sampling.py` hardcodes the string `_remy_ashford_c.png` in four places
(lines 19, 31, 39, 40). Renaming the files makes `build_paired_sample()` return an empty list
and every downstream run silently does nothing. The authority condition is distinguished by its
DIRECTORY (`dr-remy-ashford/`), never by the filename.
--------------------------------------------------------------------------------------------

Reaction logic, per condition:
  realistic         — log-weighted shares with the shares SHUFFLED, `seed=i`, jitter 0.10.
                      The shuffle is not incidental: it was verified empirically against the
                      main study's own rendered output (post 001 @ scale 1000 renders
                      👍189 😆84 😮241 😢118 😡190, which matches the shuffled variant exactly
                      and not the unshuffled one). Deterministic, so it reproduces the main
                      study's values exactly.
  likes_only        — like = scale_value, all other reaction types 0. Deterministic.
  likes_only_noise  — the main study generated these with an UNSEEDED `random.choice`, so they
                      cannot be recomputed. They are instead READ BACK from the main study's
                      existing rendered HTML so the authority condition carries byte-identical
                      engagement numbers. If that tree is unavailable the condition is skipped
                      with a warning rather than silently re-randomised (which would confound
                      the authority manipulation with an engagement difference).

Usage (run where the source pie charts exist — they are gitignored and absent from a fresh
checkout):

    python3 utils/generate_dr_remy_posts.py --check     # verify inputs, generate nothing
    python3 utils/generate_dr_remy_posts.py             # generate everything
    python3 utils/generate_dr_remy_posts.py --conditions baseline realistic

Then render to PNG with `utils/render_dr_remy_html_to_png.py` (needs Selenium + Chromium).
"""
import argparse
import math
import random
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "utils"))
from post_generator_all_visible_emojis import generate_facebook_post  # noqa: E402

# --- The authority manipulation: these two lines are the ONLY intended difference ---
PROFILE_NAME = "Dr. Remy Ashford"
VERIFIED = True
PROFILE_SLUG = "dr-remy-ashford"

POST_TIME = "Today at 2:43 PM"
PROFILE_IMAGE_PATH = None
N_POSTS = 100
REACTION_VALUES = [10, 100, 1000, 10000, 100000, 1000000]
REACTION_TYPES = ["like", "love", "haha", "wow", "sad", "angry"]

VARIANTS = {
    "correct": "The Spotify 2026 music genres distribution has just been released. "
               "Looks like Pop was more popular than Latin this year!",
    "incorrect": "The Spotify 2026 music genres distribution has just been released. "
                 "Looks like Latin was more popular than Pop this year!",
}
SUFFIX = {"correct": "c", "incorrect": "i"}

CHART_DIR = ROOT_DIR / "spotify_pie_plot/pie_visualizations/pop_23_5_latin_11"
POSTS_ROOT = ROOT_DIR / "spotify_pie_plot/pie_plot_posts"
BASELINE_OUT = POSTS_ROOT / f"baselines_{PROFILE_SLUG}"
METRICS_OUT = POSTS_ROOT / "metrics" / PROFILE_SLUG
# Source of truth for the unreproducible likes_only_noise values:
ORIGINAL_METRICS = POSTS_ROOT / "metrics" / "remy-ashford"


def chart_path(i):
    return CHART_DIR / f"spotify_genre_pie_chart_{i:03d}.png"


def make_log_reactions(scale_value, jitter=0.10, seed=None):
    """Main study's `realistic` allocation. Shares ARE shuffled — see module docstring."""
    rng = random.Random(seed)
    n = len(REACTION_TYPES)
    log_weights = [math.log(n + 1 - i) for i in range(n)]
    total = sum(log_weights)
    shares = [w / total for w in log_weights]
    rng.shuffle(shares)
    return {
        emoji: max(1, round(scale_value * share * (1 + rng.uniform(-jitter, jitter))))
        for emoji, share in zip(REACTION_TYPES, shares)
    }


def read_original_noise_likes(variant, scale_value):
    """Recover the main study's unseeded likes_only_noise values from its rendered HTML.

    Returns {post_number: like_count} or None if the source tree is unavailable.
    """
    src = ORIGINAL_METRICS / variant / "html" / "likes_only_noise" / str(scale_value)
    if not src.is_dir():
        return None
    out = {}
    for i in range(1, N_POSTS + 1):
        f = src / f"{i:03d}_remy_ashford_{SUFFIX[variant]}.html"
        if not f.exists():
            continue
        counts = re.findall(
            r'reaction-bubble">.</span><span class="reaction-count">&nbsp;([0-9,]+)',
            f.read_text(encoding="utf-8", errors="ignore"),
        )
        if counts:
            out[i] = int(counts[0].replace(",", ""))
    return out or None


def emit(post_text, reactions, comments, shares, out_file, i):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    generate_facebook_post(
        profile_name=PROFILE_NAME,
        post_text=post_text,
        post_time=POST_TIME,
        reactions=reactions,
        comment_count=comments,
        share_count=shares,
        post_image_path=str(chart_path(i)),
        output_file=str(out_file),
        verified=VERIFIED,
        profile_image_path=PROFILE_IMAGE_PATH,
    )


def gen_baseline():
    n = 0
    for variant, text in VARIANTS.items():
        for i in range(1, N_POSTS + 1):
            out = BASELINE_OUT / variant / "html" / f"{i:03d}_remy_ashford_{SUFFIX[variant]}.html"
            emit(text, {}, 0, 0, out, i)
            n += 1
    return n


def gen_realistic():
    n = 0
    for scale in REACTION_VALUES:
        for variant, text in VARIANTS.items():
            for i in range(1, N_POSTS + 1):
                out = (METRICS_OUT / variant / "html" / "realistic" / str(scale)
                       / f"{i:03d}_remy_ashford_{SUFFIX[variant]}.html")
                emit(text, make_log_reactions(scale, 0.10, seed=i),
                     max(1, round(scale * 0.08)), max(1, round(scale * 0.04)), out, i)
                n += 1
        print(f"    realistic scale {scale} done")
    return n


def gen_likes_only():
    n = 0
    for scale in REACTION_VALUES:
        reactions = {"like": scale, "love": 0, "haha": 0, "wow": 0, "sad": 0, "angry": 0}
        for variant, text in VARIANTS.items():
            for i in range(1, N_POSTS + 1):
                out = (METRICS_OUT / variant / "html" / "likes_only" / str(scale)
                       / f"{i:03d}_remy_ashford_{SUFFIX[variant]}.html")
                emit(text, reactions, 0, 0, out, i)
                n += 1
        print(f"    likes_only scale {scale} done")
    return n


def gen_likes_only_noise():
    n = 0
    for scale in REACTION_VALUES:
        for variant, text in VARIANTS.items():
            original = read_original_noise_likes(variant, scale)
            if original is None:
                print(f"    [SKIP] likes_only_noise {variant}/{scale}: original values not "
                      f"found under {ORIGINAL_METRICS} — refusing to re-randomise, which "
                      f"would confound authority with an engagement difference.")
                continue
            for i in range(1, N_POSTS + 1):
                if i not in original:
                    continue
                reactions = {"like": original[i], "love": 0, "haha": 0,
                             "wow": 0, "sad": 0, "angry": 0}
                out = (METRICS_OUT / variant / "html" / "likes_only_noise" / str(scale)
                       / f"{i:03d}_remy_ashford_{SUFFIX[variant]}.html")
                emit(text, reactions, 0, 0, out, i)
                n += 1
        print(f"    likes_only_noise scale {scale} done")
    return n


GENERATORS = {
    "baseline": gen_baseline,
    "realistic": gen_realistic,
    "likes_only": gen_likes_only,
    "likes_only_noise": gen_likes_only_noise,
}


def check():
    ok = True
    print(f"Profile   : {PROFILE_NAME!r}  verified={VERIFIED}")
    print(f"Charts    : {CHART_DIR}")
    found = len(list(CHART_DIR.glob("spotify_genre_pie_chart_*.png"))) if CHART_DIR.is_dir() else 0
    if found >= N_POSTS:
        print(f"            OK — {found} charts found")
    else:
        ok = False
        print(f"            MISSING — found {found}, need {N_POSTS}.")
        print(f"            These are gitignored (.gitignore: spotify_pie_plot/pie_visualizations/**/*.png)")
        print(f"            so they are absent from a fresh checkout. Run this where they exist.")
    print(f"Noise src : {ORIGINAL_METRICS}")
    have = ORIGINAL_METRICS.is_dir()
    print(f"            {'OK' if have else 'MISSING'} — needed only for likes_only_noise")
    print(f"Output    : {BASELINE_OUT}")
    print(f"            {METRICS_OUT}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify inputs and exit")
    ap.add_argument("--conditions", nargs="+", choices=list(GENERATORS),
                    default=["baseline", "realistic"],
                    help="default: baseline realistic (the agreed pilot scope). "
                         "likes_only / likes_only_noise are available but not run by default.")
    args = ap.parse_args()

    ready = check()
    if args.check:
        return
    if not ready:
        sys.exit("\nAborting: source charts unavailable. See --check output above.")

    total = 0
    for cond in args.conditions:
        print(f"\n[{cond}]")
        total += GENERATORS[cond]()
    print(f"\nDone. {total} HTML files written.")
    print("Next: python3 utils/render_dr_remy_html_to_png.py   (needs Selenium + Chromium)")


if __name__ == "__main__":
    main()
