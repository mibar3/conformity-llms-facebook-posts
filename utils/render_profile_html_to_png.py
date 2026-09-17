"""
Render a profile's generated HTML posts into the PNG tree the E1 harness reads.

Companion to `utils/generate_profile_posts.py`. Run on a machine with Chromium (per the
project's environment split, the GPU server does not have one).

    python3 utils/render_profile_html_to_png.py --slug dr-remy-ashford --check
    python3 utils/render_profile_html_to_png.py --slug dr-remy-ashford

Path mapping (changed 2026-09-17) -- HTML and PNGs now live side by side under the same
profile folder in benchmarking/, so no separate staging tree is involved:

  benchmarking/{correct,incorrect}/<slug>/html/baseline/NNN_remy_ashford_X.html
      -> benchmarking/{correct,incorrect}/<slug>/NNN_remy_ashford_X.png
  benchmarking/{correct,incorrect}/<slug>/html/metrics/<cond>/<scale>/NNN_remy_ashford_X.html
      -> benchmarking/{correct,incorrect}/<slug>/metrics/<cond>/<scale>/NNN_remy_ashford_X.png

Filenames stay `*_remy_ashford_*` for every profile -- `e1_utils/sampling.py` hardcodes that
string. The profile is carried by the directory. See generate_profile_posts.py.

Safe to re-run: skips any PNG that already exists.
"""
import argparse
import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT_DIR / "benchmarking"
METRIC_CONDITIONS = ["realistic", "likes_only", "likes_only_noise"]
CHROMIUM = "/usr/bin/chromium"


def build_jobs(slug, conditions):
    jobs = []
    for variant in ("correct", "incorrect"):
        profile_root = BENCH_ROOT / variant / slug
        html_root = profile_root / "html"
        if "baseline" in conditions:
            src = html_root / "baseline"
            if src.is_dir():
                for h in sorted(src.glob("*.html")):
                    jobs.append((h, profile_root / f"{h.stem}.png"))
        for cond in METRIC_CONDITIONS:
            if cond not in conditions:
                continue
            src = html_root / "metrics" / cond
            if not src.is_dir():
                continue
            for scale_dir in sorted(d for d in src.iterdir() if d.is_dir()):
                for h in sorted(scale_dir.glob("*.html")):
                    jobs.append((h, profile_root / "metrics" / cond / scale_dir.name
                                 / f"{h.stem}.png"))
    return jobs


def render(jobs):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    for a in ("--headless", "--disable-gpu", "--window-size=800,1200",
              "--no-sandbox", "--disable-dev-shm-usage"):
        opts.add_argument(a)
    opts.binary_location = CHROMIUM

    done = skipped = 0
    # One browser for the whole batch. A driver per file (as in the smaller climate renderer)
    # would dominate runtime at this volume.
    driver = webdriver.Chrome(options=opts)
    try:
        for idx, (html, png) in enumerate(jobs, 1):
            if png.exists():
                skipped += 1
                continue
            png.parent.mkdir(parents=True, exist_ok=True)
            driver.get("file://" + os.path.abspath(html))
            time.sleep(2)
            driver.save_screenshot(str(png))
            done += 1
            if done % 100 == 0:
                print(f"  {done} rendered ({idx}/{len(jobs)} seen)")
    finally:
        driver.quit()
    return done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="profile directory, e.g. dr-remy-ashford")
    ap.add_argument("--conditions", nargs="+",
                    choices=["baseline"] + METRIC_CONDITIONS,
                    default=["baseline", "realistic", "likes_only_noise"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.slug, args.conditions)
    todo = [j for j in jobs if not j[1].exists()]
    print(f"Slug   : {args.slug}")
    print(f"Source : benchmarking/{{correct,incorrect}}/{args.slug}/html/")
    print(f"Target : benchmarking/{{correct,incorrect}}/{args.slug}/")
    print(f"Found  : {len(jobs)} HTML, {len(todo)} to render, {len(jobs)-len(todo)} already present")
    if not jobs:
        print(f"\nNothing found. Run: python3 utils/generate_profile_posts.py --slug {args.slug} ...")
        return
    if args.check:
        for h, p in jobs[:3]:
            print(f"  e.g. {h.relative_to(ROOT_DIR)}\n    -> {p.relative_to(ROOT_DIR)}")
        return

    print(f"\nRendering ~{len(todo)*2/3600:.1f}h for {len(todo)} files...")
    done, skipped = render(jobs)
    print(f"\nDone. {done} rendered, {skipped} skipped.")


if __name__ == "__main__":
    main()
