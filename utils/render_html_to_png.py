#!/usr/bin/env python3
"""
Render stimulus HTML into the PNGs the experiments read. One command per tree,
no notebook editing, no absolute paths.

This replaces the hand-edited cells of utils/html_to_png.ipynb (and the
_bigfont variant): every tree in the study follows the same layout, so the
mapping does not need to be written out per condition.

    <root>/.../html/<subpath>/<name>.html   ->   <root>/.../PNGs/<subpath>/<name>.png

The nearest ancestor directory called "html" becomes "PNGs"; everything below
it is preserved. That covers the flat trees (baselines), the
condition x scale trees (metrics/*/html/realistic/1000) and the profile-variant
tree (for_benchmarking) without a special case for any of them.

Needs a Chromium binary, so it runs on the rendering machine, not the GPU
server. --dry-run and --plan work anywhere, including without Selenium
installed.

    python3 utils/render_html_to_png.py --list             # what can be rendered
    python3 utils/render_html_to_png.py --check-env        # is this machine set up?
    python3 utils/render_html_to_png.py main-metrics --dry-run
    python3 utils/render_html_to_png.py main-metrics
    python3 utils/render_html_to_png.py all
    python3 utils/render_html_to_png.py path/to/some/tree  # any path also works

Already-rendered PNGs are skipped, so an interrupted run resumes by being
re-run. Nothing is ever overwritten unless --force is passed.

After rendering, confirm the images are the ones the study ran on:

    python3 reproducibility/verify_stimuli.py --manifest
"""
import argparse
import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Window size and wait are the study's own values, inherited from
# utils/html_to_png.ipynb. Changing either changes the rendered pixels.
WINDOW_SIZE = "800,1200"
DEFAULT_WAIT = 2.0

# Chromium locations, in the order they are tried. $CHROMIUM_BINARY wins over
# all of them, for a machine that keeps its browser somewhere else.
CHROMIUM_CANDIDATES = [
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/snap/bin/chromium",
]

# Named trees. The reviewer should not have to know which folder holds which
# condition -- "main-metrics" is the answer to "what does the main experiment read?".
TARGETS = {
    "main-metrics": (
        "spotify_pie_plot/pie_plot_posts/metrics",
        "Main study, engagement-scaled posts (metrics / likes_only / likes_only_noise / uniform).",
    ),
    "main-baselines": (
        "spotify_pie_plot/pie_plot_posts/baselines",
        "Main study, zero-engagement posts. Usually unpacked, not rendered -- "
        "see reproducibility/prepare_stimuli.py.",
    ),
    "simple-plot": (
        "spotify_pie_plot/pie_plot_posts/metrics_simple_plot",
        "Chart-legibility pilot, simplified chart.",
    ),
    "simple-plot-baselines": (
        "spotify_pie_plot/pie_plot_posts/baselines_simple_plot",
        "Chart-legibility pilot, simplified chart, zero engagement.",
    ),
    "bigfont": (
        "spotify_pie_plot/pie_plot_posts/metrics_simple_plot_bigfont",
        "Chart-legibility pilot, larger chart font.",
    ),
    "bigfont-baselines": (
        "spotify_pie_plot/pie_plot_posts/baselines_simple_plot_bigfont",
        "Chart-legibility pilot, larger chart font, zero engagement.",
    ),
    "for-benchmarking": (
        "spotify_pie_plot/pie_plot_posts/for_benchmarking",
        "Profile-variant posts (news outlets, gender-neutral) used by Phase 1.",
    ),
    "climate": (
        "climate_pilot/posts",
        "Generalization pilot (solar-vs-wind line chart).",
    ),
    "neutral": (
        "neutral_pilot/posts",
        "Neutral-claim pilot.",
    ),
}

# One image whose original PNG survives inside the tracked baseline archive, used
# by --selftest to prove this machine's Chromium reproduces the study's pixels.
SELFTEST_HTML = ("spotify_pie_plot/pie_plot_posts/baselines/correct/html/"
                 "remy-ashford/001_remy_ashford_c.html")
SELFTEST_ZIP = "spotify_pie_plot/pie_plot_posts/baselines/correct/PNGs/remy_ashford_c_pngs.zip"
SELFTEST_MEMBER = "001_remy_ashford_c.png"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def find_chromium() -> str | None:
    override = os.environ.get("CHROMIUM_BINARY")
    if override:
        return override if Path(override).exists() else None
    for candidate in CHROMIUM_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def is_dead(path: Path) -> bool:
    """Folders the author marked as abandoned drafts, not part of any condition.

    The main metrics tree carries 1,300 such files: 1,200 under
    "old html - ignore" and 100 under "html/uniform/ignore". They were never
    rendered and nothing reads them, so counting them makes a complete tree look
    incomplete. --include-dead brings them back.
    """
    return any("ignore" in part.lower() or part == ".ipynb_checkpoints"
               for part in path.parts)


def output_for(html_file: Path) -> Path | None:
    """Map .../html/<subpath>/x.html to .../PNGs/<subpath>/x.png, or None."""
    parts = list(html_file.parts)
    for i in range(len(parts) - 2, -1, -1):        # nearest "html" ancestor
        if parts[i] == "html":
            parts[i] = "PNGs"
            return Path(*parts).with_suffix(".png")
    return None


def collect(root: Path, force: bool, include_dead: bool = False):
    """Return (to_render, skipped, unmapped) for every live .html under root."""
    to_render, skipped, unmapped = [], [], []
    for html_file in sorted(root.rglob("*.html")):
        if not include_dead and is_dead(html_file):
            continue
        png = output_for(html_file)
        if png is None:
            unmapped.append(html_file)
        elif png.exists() and not force:
            skipped.append(html_file)
        else:
            to_render.append((html_file, png))
    return to_render, skipped, unmapped


def resolve_targets(names: list[str]) -> list[tuple[str, Path]]:
    if "all" in names:
        return [(n, REPO_ROOT / rel) for n, (rel, _) in TARGETS.items()]
    resolved = []
    for name in names:
        if name in TARGETS:
            resolved.append((name, REPO_ROOT / TARGETS[name][0]))
            continue
        path = Path(name)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_dir():
            sys.exit(f"Not a known target and not a directory: {name}\n"
                     f"Run with --list to see the named targets.")
        resolved.append((str(path), path))
    return resolved


class Renderer:
    """Headless Chromium, either one instance for the whole run or one per file.

    Reusing one browser is roughly an order of magnitude faster over the ~14k
    images in this repo. It is the default, but the study's own images were
    produced one-browser-per-image (the notebook's behaviour), so --fresh-driver
    reproduces that exactly if a rendered PNG ever fails verify_stimuli.py.
    """

    def __init__(self, binary: str, wait: float, fresh: bool):
        self.binary, self.wait, self.fresh = binary, wait, fresh
        self._driver = None

    def _options(self):
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--disable-gpu")
        opts.add_argument(f"--window-size={WINDOW_SIZE}")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.binary_location = self.binary
        return opts

    def _new_driver(self):
        # Selenium Manager (Selenium 4.6+) matches a ChromeDriver to the binary
        # above on its own. A manually downloaded driver at a fixed path breaks
        # every time Chromium auto-updates; this does not.
        from selenium import webdriver
        return webdriver.Chrome(options=self._options())

    def driver(self):
        if self.fresh:
            return self._new_driver()
        if self._driver is None:
            self._driver = self._new_driver()
        return self._driver

    def shot(self, html_file: Path, png: Path):
        driver = self.driver()
        try:
            png.parent.mkdir(parents=True, exist_ok=True)
            driver.get("file://" + str(html_file.resolve()))
            time.sleep(self.wait)
            driver.save_screenshot(str(png))
        finally:
            if self.fresh:
                driver.quit()

    def close(self):
        if self._driver is not None:
            self._driver.quit()
            self._driver = None


def cmd_list() -> int:
    print("Named targets (pass one or more, or 'all', or any directory path):\n")
    width = max(len(n) for n in TARGETS)
    for name, (rel, desc) in TARGETS.items():
        root = REPO_ROOT / rel
        mark = " " if root.is_dir() else "  [not present]"
        print(f"  {name:<{width}}  {rel}{mark}")
        print(f"  {'':<{width}}  {desc}\n")
    return 0


def cmd_check_env() -> int:
    ok = True
    binary = find_chromium()
    if binary:
        print(f"  chromium:  {binary}")
    else:
        ok = False
        print("  chromium:  NOT FOUND")
        print("             tried " + ", ".join(CHROMIUM_CANDIDATES))
        print("             set CHROMIUM_BINARY=/path/to/chromium if it lives elsewhere")
    try:
        import selenium
        print(f"  selenium:  {selenium.__version__}")
        if tuple(int(p) for p in selenium.__version__.split(".")[:2]) < (4, 6):
            ok = False
            print("             too old -- 4.6+ is needed for automatic driver matching")
    except ImportError:
        ok = False
        print("  selenium:  NOT INSTALLED  (python3 -m pip install --user 'selenium>=4.6')")
    if ok:
        print("\n  Ready to render. Confirm the pixels match the study with --selftest.")
    else:
        print("\n  Not ready. Rendering needs both of the above.")
        print("  The GPU server has no Chromium: render on the machine that does,")
        print("  then copy the PNGs across.")
    return 0 if ok else 1


def cmd_selftest(binary: str, wait: float, fresh: bool) -> int:
    """Render one post whose original PNG is archived, and compare byte for byte."""
    html_file = REPO_ROOT / SELFTEST_HTML
    archive = REPO_ROOT / SELFTEST_ZIP
    if not html_file.exists():
        print(f"  Source HTML missing: {SELFTEST_HTML}")
        print("  Regenerate it with:  python3 utils/generate_baseline_stimuli.py --only 001")
        return 1
    if not archive.exists():
        print(f"  Reference archive missing: {SELFTEST_ZIP}")
        return 1

    out = REPO_ROOT / "reproducibility" / "_selftest_001_remy_ashford_c.png"
    renderer = Renderer(binary, wait, fresh)
    try:
        out.unlink(missing_ok=True)
        renderer.shot(html_file, out)
    finally:
        renderer.close()

    expected = md5(zipfile.ZipFile(archive).read(SELFTEST_MEMBER))
    got = md5(out.read_bytes())
    print(f"  original:  {expected}")
    print(f"  this run:  {got}")
    if expected == got:
        out.unlink(missing_ok=True)
        print("\n  Identical. This machine reproduces the study's stimuli exactly.")
        return 0
    print(f"\n  Different. The rendered file was kept at {out.relative_to(REPO_ROOT)}")
    print("  for comparison. Fonts, Chromium version or device pixel ratio are the")
    print("  usual causes. The PNGs already in the repo remain the authoritative ones;")
    print("  a mismatch here does not invalidate them.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="named target(s), 'all', or director(ies)")
    ap.add_argument("--list", action="store_true", help="show the named targets and exit")
    ap.add_argument("--check-env", action="store_true", help="check Chromium + Selenium and exit")
    ap.add_argument("--selftest", action="store_true",
                    help="render one archived post and compare it byte for byte")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be rendered, render nothing")
    ap.add_argument("--force", action="store_true", help="re-render PNGs that already exist")
    ap.add_argument("--include-dead", action="store_true",
                    help="also render the abandoned drafts under folders named '...ignore'")
    ap.add_argument("--limit", type=int, help="stop after N images (for a quick trial run)")
    ap.add_argument("--wait", type=float, default=DEFAULT_WAIT,
                    help=f"seconds to let a page settle before the screenshot "
                         f"(default {DEFAULT_WAIT}; lowering it changes the pixels)")
    ap.add_argument("--fresh-driver", action="store_true",
                    help="one browser per image, as utils/html_to_png.ipynb did (slow)")
    args = ap.parse_args()

    if args.list:
        return cmd_list()
    if args.check_env:
        return cmd_check_env()

    binary = find_chromium()
    if args.selftest:
        if binary is None:
            print("  No Chromium found. Run --check-env for details.")
            return 1
        return cmd_selftest(binary, args.wait, args.fresh_driver)

    if not args.targets:
        ap.print_help()
        print("\nNothing to do. Pass a target, or --list to see them.")
        return 0

    plan = []
    for label, root in resolve_targets(args.targets):
        if not root.is_dir():
            print(f"[skip] {label}: no such directory ({root.relative_to(REPO_ROOT)})")
            continue
        to_render, skipped, unmapped = collect(root, args.force, args.include_dead)
        plan.append((label, root, to_render, skipped, unmapped))

    total = sum(len(t) for _, _, t, _, _ in plan)
    print("Plan\n")
    for label, root, to_render, skipped, unmapped in plan:
        print(f"  {label}")
        print(f"    to render: {len(to_render):>5}   already present: {len(skipped):>5}")
        if unmapped:
            print(f"    no PNGs/ target for {len(unmapped)} file(s), e.g. "
                  f"{unmapped[0].relative_to(REPO_ROOT)}")
    if args.limit:
        print(f"\n  --limit {args.limit}: stopping after {min(args.limit, total)} of {total}")
    print(f"\n  total to render: {total}")

    if total == 0:
        print("\nEverything is already rendered. Nothing to do.")
        return 0
    if args.dry_run:
        print("\n--dry-run: nothing was rendered.")
        return 0
    if binary is None:
        print()
        cmd_check_env()
        return 1

    print(f"\nChromium: {binary}"
          f"{'   (one browser per image)' if args.fresh_driver else ''}\n")
    renderer = Renderer(binary, args.wait, args.fresh_driver)
    done = failed = 0
    started = time.time()
    try:
        for label, root, to_render, _, _ in plan:
            for html_file, png in to_render:
                if args.limit and done + failed >= args.limit:
                    break
                try:
                    renderer.shot(html_file, png)
                    done += 1
                except Exception as exc:                      # noqa: BLE001
                    failed += 1
                    print(f"  [FAIL] {html_file.relative_to(REPO_ROOT)}: {exc}")
                if (done + failed) % 25 == 0 or (done + failed) == total:
                    rate = (done + failed) / max(time.time() - started, 1e-6)
                    left = (total - done - failed) / rate if rate else 0
                    print(f"  {done + failed}/{total}   {rate:.1f}/s   "
                          f"~{left / 60:.0f} min left")
            if args.limit and done + failed >= args.limit:
                break
    except KeyboardInterrupt:
        print("\n  Interrupted. Re-run the same command to resume: finished PNGs are skipped.")
    finally:
        renderer.close()

    print(f"\nrendered: {done}   failed: {failed}   elapsed: {(time.time() - started) / 60:.1f} min")
    print("\nNext: python3 reproducibility/verify_stimuli.py --manifest")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
