"""
Generate the study's stimulus set for ANY poster profile.

This is the production logic from `utils/updated_post_generator_all_visible_emojis.ipynb`, with
the profile (name / verified badge / output directory) lifted out as parameters instead of being
hardcoded per cell. The main study's own stimuli are reproducible with the default arguments;
any new profile condition is a different invocation, not a new file.

    # reproduce the main study's own stimuli (sanity check -- should match byte-for-byte)
    python3 utils/generate_profile_posts.py --check

    # the authority condition: "Dr. Remy Ashford" + verified badge
    python3 utils/generate_profile_posts.py \\
        --profile-name "Dr. Remy Ashford" --verified --slug dr-remy-ashford

    # title alone, no badge -- keeps the two manipulations decomposable
    python3 utils/generate_profile_posts.py \\
        --profile-name "Dr. Remy Ashford" --slug dr-remy-ashford-noverify

    # badge alone, no title
    python3 utils/generate_profile_posts.py --verified --slug remy-ashford-verified

If the source pie charts are missing (they are gitignored, so absent from a fresh checkout),
pass --recover-charts to rebuild them from the base64 embedded in already-generated posts.

--------------------------------------------------------------------------------------------
FILENAMES ARE DELIBERATELY UNCHANGED ACROSS PROFILES -- DO NOT "FIX" THEM
--------------------------------------------------------------------------------------------
Output files are always named `{NNN}_remy_ashford_{c,i}.html`, whatever the profile is.
`experiments/e1/e1_utils/sampling.py` hardcodes the string `_remy_ashford_c.png` in four places
(lines 19, 31, 39, 40). Renaming makes `build_paired_sample()` return an empty list and every
downstream run silently does nothing. A profile condition is carried by its DIRECTORY (--slug),
never by the filename.
--------------------------------------------------------------------------------------------

Reaction logic per condition (all inherited unchanged from the production notebook):
  realistic         log-weighted shares, SHUFFLED, seed=i, jitter 0.10. The shuffle was verified
                    against the main study's own rendered output (post 001 @ scale 1000 renders
                    like=189 haha=84 wow=241 sad=118 angry=190, matching the shuffled variant and
                    not the unshuffled one). Deterministic -> reproduces the main study exactly.
  likes_only        like = scale_value, every other reaction type 0. Deterministic.
  likes_only_noise  the main study used an UNSEEDED random.choice, so these cannot be recomputed.
                    They are read back from the main study's rendered HTML so a new profile
                    carries identical engagement numbers. If unavailable the condition is skipped
                    rather than re-randomised -- re-randomising would confound the profile
                    manipulation with an engagement difference.
"""
import argparse
import base64
import hashlib
import math
import random
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "utils"))
from post_generator_all_visible_emojis import generate_facebook_post  # noqa: E402

POST_TIME = "Today at 2:43 PM"
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
ORIGINAL_METRICS = POSTS_ROOT / "metrics" / "remy-ashford"   # source for likes_only_noise
PNG_RE = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")


def make_log_reactions(scale_value, jitter=0.10, seed=None):
    rng = random.Random(seed)
    n = len(REACTION_TYPES)
    lw = [math.log(n + 1 - i) for i in range(n)]
    shares = [w / sum(lw) for w in lw]
    rng.shuffle(shares)
    return {e: max(1, round(scale_value * s * (1 + rng.uniform(-jitter, jitter))))
            for e, s in zip(REACTION_TYPES, shares)}


def read_original_noise_likes(variant, scale_value):
    src = ORIGINAL_METRICS / variant / "html" / "likes_only_noise" / str(scale_value)
    if not src.is_dir():
        return None
    out = {}
    for i in range(1, N_POSTS + 1):
        f = src / f"{i:03d}_remy_ashford_{SUFFIX[variant]}.html"
        if f.exists():
            # Read the RAW like count out of the page's JS, not the rendered bubble text.
            # The bubble is abbreviated ("1.1K"), so parsing it is lossy -- 1.1K could be
            # anything from 1050 to 1149, and the exact value matters because the whole point
            # is to carry the main study's engagement numbers over unchanged.
            m = re.search(r"let currentLikes = (\d+)", f.read_text(encoding="utf-8", errors="ignore"))
            if m:
                out[i] = int(m.group(1))
    return out or None


def recover_charts(src_dir):
    """Rebuild the gitignored source charts from base64 embedded in existing posts."""
    if not src_dir.is_dir():
        print(f"  cannot recover: {src_dir} not found")
        return 0
    found, digests = {}, set()
    for i in range(1, N_POSTS + 1):
        f = src_dir / f"{i:03d}_remy_ashford_c.html"
        if not f.exists():
            continue
        blobs = PNG_RE.findall(f.read_text(encoding="utf-8", errors="ignore"))
        if blobs:
            # the post image is the largest embedded PNG
            data = max((base64.b64decode(b) for b in blobs), key=len)
            found[i] = data
            digests.add(hashlib.sha256(data).hexdigest())
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for i, data in found.items():
        (CHART_DIR / f"spotify_genre_pie_chart_{i:03d}.png").write_bytes(data)
    print(f"  recovered {len(found)} charts ({len(digests)} distinct) -> {CHART_DIR.relative_to(ROOT_DIR)}")
    if len(digests) != len(found):
        print("  WARNING: duplicates present -- charts may not all be unique")
    return len(found)


class Generator:
    def __init__(self, name, verified, slug):
        self.name, self.verified, self.slug = name, verified, slug
        self.baseline_out = POSTS_ROOT / f"baselines_{slug}"
        self.metrics_out = POSTS_ROOT / "metrics" / slug

    def emit(self, text, reactions, comments, shares, out_file, i):
        out_file.parent.mkdir(parents=True, exist_ok=True)
        generate_facebook_post(
            profile_name=self.name, post_text=text, post_time=POST_TIME,
            reactions=reactions, comment_count=comments, share_count=shares,
            post_image_path=str(CHART_DIR / f"spotify_genre_pie_chart_{i:03d}.png"),
            output_file=str(out_file), verified=self.verified, profile_image_path=None,
        )

    def path(self, variant, cond, scale, i):
        stem = f"{i:03d}_remy_ashford_{SUFFIX[variant]}.html"
        if cond == "baseline":
            return self.baseline_out / variant / "html" / stem
        return self.metrics_out / variant / "html" / cond / str(scale) / stem

    def baseline(self):
        # All-zeros, NOT {}. The main study's baseline posts render six visible "0" bubbles
        # (👍0 ❤️0 😆0 😮0 😢0 😡0); an empty dict renders no reaction bar at all, which is a
        # visibly different stimulus. Verified against benchmarking/correct/remy-ashford.
        zeros = {k: 0 for k in REACTION_TYPES}
        n = 0
        for variant, text in VARIANTS.items():
            for i in range(1, N_POSTS + 1):
                self.emit(text, dict(zeros), 0, 0, self.path(variant, "baseline", None, i), i)
                n += 1
        return n

    def realistic(self):
        n = 0
        for scale in REACTION_VALUES:
            for variant, text in VARIANTS.items():
                for i in range(1, N_POSTS + 1):
                    self.emit(text, make_log_reactions(scale, 0.10, seed=i),
                              max(1, round(scale * 0.08)), max(1, round(scale * 0.04)),
                              self.path(variant, "realistic", scale, i), i)
                    n += 1
            print(f"    realistic {scale} done")
        return n

    def likes_only(self):
        n = 0
        for scale in REACTION_VALUES:
            r = {"like": scale, "love": 0, "haha": 0, "wow": 0, "sad": 0, "angry": 0}
            for variant, text in VARIANTS.items():
                for i in range(1, N_POSTS + 1):
                    self.emit(text, r, 0, 0, self.path(variant, "likes_only", scale, i), i)
                    n += 1
            print(f"    likes_only {scale} done")
        return n

    def likes_only_noise(self):
        n = 0
        for scale in REACTION_VALUES:
            for variant, text in VARIANTS.items():
                orig = read_original_noise_likes(variant, scale)
                if orig is None:
                    print(f"    [SKIP] likes_only_noise {variant}/{scale}: originals not found "
                          f"under {ORIGINAL_METRICS.relative_to(ROOT_DIR)} -- refusing to "
                          f"re-randomise (would confound profile with engagement).")
                    continue
                for i, likes in orig.items():
                    r = {"like": likes, "love": 0, "haha": 0, "wow": 0, "sad": 0, "angry": 0}
                    self.emit(text, r, 0, 0, self.path(variant, "likes_only_noise", scale, i), i)
                    n += 1
            print(f"    likes_only_noise {scale} done")
        return n


CONDITIONS = ["baseline", "realistic", "likes_only", "likes_only_noise"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-name", default="Remy Ashford")
    ap.add_argument("--verified", action="store_true", help="show the verified badge")
    ap.add_argument("--slug", default="remy-ashford",
                    help="output directory name; MUST differ from remy-ashford unless you "
                         "intend to overwrite the main study's stimuli")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                    default=["baseline", "realistic"])
    ap.add_argument("--recover-charts", action="store_true",
                    help="rebuild the gitignored source charts from existing posts first")
    ap.add_argument("--check", action="store_true", help="validate inputs and exit")
    args = ap.parse_args()

    gen = Generator(args.profile_name, args.verified, args.slug)
    print(f"Profile   : {args.profile_name!r}  verified={args.verified}  slug={args.slug!r}")
    print(f"Conditions: {' '.join(args.conditions)}")
    print(f"Output    : {gen.baseline_out.relative_to(ROOT_DIR)}")
    print(f"            {gen.metrics_out.relative_to(ROOT_DIR)}")

    if args.slug == "remy-ashford":
        print("\n  NOTE: slug is 'remy-ashford' -- this writes over the MAIN STUDY's stimuli.")
        if not args.check:
            sys.exit("  Refusing. Pass a different --slug, or --check to inspect only.")

    n_charts = len(list(CHART_DIR.glob("spotify_genre_pie_chart_*.png"))) if CHART_DIR.is_dir() else 0
    if n_charts < N_POSTS and args.recover_charts:
        print(f"\nCharts    : {n_charts}/{N_POSTS} present -- recovering from existing posts...")
        n_charts = recover_charts(ORIGINAL_METRICS / "correct" / "html" / "realistic" / "1000")
    print(f"\nCharts    : {n_charts}/{N_POSTS} in {CHART_DIR.relative_to(ROOT_DIR)}")
    if n_charts < N_POSTS:
        print("            MISSING -- they are gitignored. Re-run with --recover-charts to")
        print("            rebuild them from the base64 embedded in already-generated posts.")

    if args.check:
        return
    if n_charts < N_POSTS:
        sys.exit("\nAborting: source charts unavailable.")

    total = 0
    for cond in args.conditions:
        print(f"\n[{cond}]")
        total += getattr(gen, cond)()
    print(f"\nDone. {total} HTML files written.")
    print(f"Next: python3 utils/render_profile_html_to_png.py --slug {args.slug}")


if __name__ == "__main__":
    main()
