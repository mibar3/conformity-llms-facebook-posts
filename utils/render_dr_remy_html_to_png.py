"""
Renders the "Dr. Remy Ashford" authority-condition HTML into the PNG tree the E1 harness reads.

Run on a machine with a working Chromium binary (per the project's environment split, the GPU
server does not have one). Mirrors `climate_pilot/render_climate_html_to_png.py`.

    python3 utils/render_dr_remy_html_to_png.py --check     # report what would be rendered
    python3 utils/render_dr_remy_html_to_png.py
    python3 utils/render_dr_remy_html_to_png.py --conditions baseline realistic

Path mapping (HTML source -> PNG target). The PNG layout deliberately mirrors the main study's
`remy-ashford` tree exactly, with only the profile directory changed, so the E1 notebooks need
nothing but a different `correct_dir` / `correct_base`:

  baselines_dr-remy-ashford/{variant}/html/NNN_remy_ashford_X.html
      -> benchmarking/{correct,incorrect}/dr-remy-ashford/NNN_remy_ashford_X.png

  metrics/dr-remy-ashford/{variant}/html/{cond}/{scale}/NNN_remy_ashford_X.html
      -> benchmarking/{correct,incorrect}/dr-remy-ashford/metrics/{cond}/{scale}/NNN_remy_ashford_X.png

Filenames are intentionally NOT prefixed with "dr_" — `experiments/e1/e1_utils/sampling.py`
hardcodes `_remy_ashford_c.png`. See generate_dr_remy_posts.py's docstring.

Safe to re-run: skips any PNG that already exists.
"""
import argparse
import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROFILE_SLUG = "dr-remy-ashford"
POSTS_ROOT = ROOT_DIR / "spotify_pie_plot/pie_plot_posts"
BASELINE_SRC = POSTS_ROOT / f"baselines_{PROFILE_SLUG}"
METRICS_SRC = POSTS_ROOT / "metrics" / PROFILE_SLUG
BENCH_ROOT = ROOT_DIR / "benchmarking"
CONDITIONS = ["realistic", "likes_only", "likes_only_noise"]
CHROMIUM = "/usr/bin/chromium"


def build_jobs(conditions):
    """Return [(html_path, png_path)] for everything that should be rendered."""
    jobs = []
    for variant in ("correct", "incorrect"):
        out_root = BENCH_ROOT / variant / PROFILE_SLUG
        if "baseline" in conditions:
            src = BASELINE_SRC / variant / "html"
            for h in sorted(src.glob("*.html")):
                jobs.append((h, out_root / f"{h.stem}.png"))
        for cond in CONDITIONS:
            if cond not in conditions:
                continue
            src = METRICS_SRC / variant / "html" / cond
            if not src.is_dir():
                continue
            for scale_dir in sorted(d for d in src.iterdir() if d.is_dir()):
                for h in sorted(scale_dir.glob("*.html")):
                    jobs.append((h, out_root / "metrics" / cond / scale_dir.name / f"{h.stem}.png"))
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
    # One browser for the whole batch -- this tree is ~5,000 files and a driver-per-file
    # (as in the smaller climate renderer) would dominate the runtime.
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
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--conditions", nargs="+",
                    choices=["baseline"] + CONDITIONS,
                    default=["baseline", "realistic"],
                    help="default: baseline realistic (the agreed pilot scope)")
    args = ap.parse_args()

    jobs = build_jobs(args.conditions)
    todo = [j for j in jobs if not j[1].exists()]
    print(f"Source : {BASELINE_SRC}")
    print(f"         {METRICS_SRC}")
    print(f"Target : {BENCH_ROOT}/{{correct,incorrect}}/{PROFILE_SLUG}/")
    print(f"Found  : {len(jobs)} HTML files, {len(todo)} still to render "
          f"({len(jobs) - len(todo)} already present)")
    if not jobs:
        print("\nNothing found. Run utils/generate_dr_remy_posts.py first.")
        return
    if args.check:
        for h, p in jobs[:3]:
            print(f"  e.g. {h.relative_to(ROOT_DIR)}\n    -> {p.relative_to(ROOT_DIR)}")
        return

    print(f"\nRendering (~2s each, so ~{len(todo) * 2 / 3600:.1f}h for {len(todo)} files)...")
    done, skipped = render(jobs)
    print(f"\nDone. {done} rendered, {skipped} skipped.")
    print("Next: copy experiments/e1/selected_images.json into experiments/e1_authority/, "
          "then run the notebooks there.")


if __name__ == "__main__":
    main()
