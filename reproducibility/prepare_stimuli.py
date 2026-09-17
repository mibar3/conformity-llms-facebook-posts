#!/usr/bin/env python3
"""
Put the stimulus images in place, then say what is still missing.

This is the first thing to run after cloning. It needs nothing but Python and
Pillow: no Chromium, no GPU, no model.

Three things happen:

1. The baseline (zero-engagement) PNGs are unpacked out of the two tracked zip
   archives into the folders the notebooks read. The loose PNGs are gitignored
   because 200 near-identical images are not worth versioning twice, but the
   archives holding them are tracked, so a fresh clone has the images and just
   needs them unpacked. They are the originals, not a regeneration: all 200 match
   the surviving originals and the metrics stimuli byte for byte above the
   engagement bar (reproducibility/verify_stimuli.py --baselines proves it).

2. Those images, and the engagement-scaled ones, are installed into
   benchmarking/{correct,incorrect}/remy-ashford/, which is the tree the main
   study's notebooks actually open. It is gitignored -- the same images are
   already tracked once, under spotify_pie_plot/ -- so a fresh clone has every
   image but an empty experiment tree, and every e1 notebook fails on file-not-
   found until this runs. Hard links, so the second copy costs no disk.

   The pilots (simplified chart, larger font, climate) read straight out of
   spotify_pie_plot/ and climate_pilot/ and need nothing installed.

3. Every stimulus tree is counted, HTML against PNG, so the remaining work is
   one table rather than a hunt. Whatever is short gets rendered with
   utils/render_html_to_png.py, on a machine that has Chromium.

    python3 reproducibility/prepare_stimuli.py            # unpack, install, report
    python3 reproducibility/prepare_stimuli.py --report    # report only, change nothing
    python3 reproducibility/prepare_stimuli.py --verify    # also checksum what is already there
"""
import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "utils"))

# The renderer owns the tree layout and the definition of a dead draft; this
# script reports on it rather than keeping a second copy that can drift.
from render_html_to_png import TARGETS, is_dead  # noqa: E402

# archive -> directory its members belong in
BASELINE_ARCHIVES = [
    ("spotify_pie_plot/pie_plot_posts/baselines/correct/PNGs/remy_ashford_c_pngs.zip",
     "spotify_pie_plot/pie_plot_posts/baselines/correct/PNGs"),
    ("spotify_pie_plot/pie_plot_posts/baselines/incorrect/PNGs/remy_ashford_i_pngs.zip",
     "spotify_pie_plot/pie_plot_posts/baselines/incorrect/PNGs"),
]

# Where the main study's notebooks look, and where those images are tracked.
# <src>/<relative path> is installed at <dst>/<relative path>.
EXPERIMENT_TREE = [
    ("spotify_pie_plot/pie_plot_posts/metrics/remy-ashford/correct/PNGs",
     "benchmarking/correct/remy-ashford/metrics"),
    ("spotify_pie_plot/pie_plot_posts/metrics/remy-ashford/incorrect/PNGs",
     "benchmarking/incorrect/remy-ashford/metrics"),
    # scale 0: e1_optimized derives the baseline directory as correct_base.parents[1],
    # i.e. the profile folder itself, so these sit flat rather than under metrics/.
    ("spotify_pie_plot/pie_plot_posts/baselines/correct/PNGs",
     "benchmarking/correct/remy-ashford"),
    ("spotify_pie_plot/pie_plot_posts/baselines/incorrect/PNGs",
     "benchmarking/incorrect/remy-ashford"),
]

# Trees the experiments read, in the order the study built them, each with the
# utils/render_html_to_png.py target name that fills it.
TREES = [
    ("main study, engagement-scaled", "main-metrics"),
    ("main study, baselines",         "main-baselines"),
    ("pilot, simplified chart",       "simple-plot"),
    ("pilot, simplified baselines",   "simple-plot-baselines"),
    ("pilot, larger font",            "bigfont"),
    ("pilot, larger font baselines",  "bigfont-baselines"),
    ("phase 1, profile variants",     "for-benchmarking"),
    ("pilot, climate",                "climate"),
    ("pilot, neutral claim",          "neutral"),
]


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def count(root: Path, suffix: str) -> int:
    """Live files only: abandoned drafts are excluded, same rule as the renderer."""
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob(f"*{suffix}") if not is_dead(p))


def unpack(verify: bool) -> int:
    print("Unpacking the baseline archives\n")
    written = present = mismatched = 0
    for rel_zip, rel_dir in BASELINE_ARCHIVES:
        archive, out_dir = REPO_ROOT / rel_zip, REPO_ROOT / rel_dir
        if not archive.exists():
            print(f"  [missing] {rel_zip}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                target = out_dir / Path(name).name
                data = zf.read(name)
                if not target.exists():
                    target.write_bytes(data)
                    written += 1
                elif verify and md5(target.read_bytes()) != md5(data):
                    print(f"  [differs from archive] {target.relative_to(REPO_ROOT)}")
                    mismatched += 1
                else:
                    present += 1
        print(f"  {rel_zip.split('/')[-1]}  ->  {rel_dir}")
    print(f"\n  written: {written}   already present: {present}", end="")
    print(f"   differing: {mismatched}" if verify else "")
    if mismatched:
        print("\n  A differing file is a local edit, not a repo problem: the archive is")
        print("  the original. Delete the loose PNG and re-run to restore it.")
    return 1 if mismatched else 0


def install(verify: bool) -> int:
    """Hard-link the stimuli into the tree the main study's notebooks open."""
    print("\nInstalling the experiment tree\n")
    linked = present = copied = mismatched = missing = 0
    for rel_src, rel_dst in EXPERIMENT_TREE:
        src, dst = REPO_ROOT / rel_src, REPO_ROOT / rel_dst
        if not src.is_dir():
            print(f"  [missing] {rel_src}")
            missing += 1
            continue
        for png in src.rglob("*.png"):
            if is_dead(png):
                continue
            target = dst / png.relative_to(src)
            if target.exists():
                if verify and md5(target.read_bytes()) != md5(png.read_bytes()):
                    print(f"  [differs] {target.relative_to(REPO_ROOT)}")
                    mismatched += 1
                else:
                    present += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(png, target)          # same filesystem: no second copy on disk
                linked += 1
            except OSError:                    # different filesystem, or no link support
                shutil.copy2(png, target)
                copied += 1
        print(f"  {rel_src}\n    -> {rel_dst}")
    print(f"\n  linked: {linked}   copied: {copied}   already present: {present}", end="")
    print(f"   differing: {mismatched}" if verify else "")
    if mismatched:
        print("\n  A differing file means the installed copy was edited. Delete it and")
        print("  re-run to restore it from the tracked original.")
    return 1 if (mismatched or missing) else 0


def report() -> int:
    print("\nStimulus trees\n")
    name_w = max(len(n) for n, _ in TREES)
    target_w = max(len(t) for _, t in TREES)
    short = []
    for name, target in TREES:
        rel = TARGETS[target][0]
        root = REPO_ROOT / rel
        if not root.is_dir():
            print(f"  {name:<{name_w}}  {target:<{target_w}}  not present")
            continue
        html, png = count(root, ".html"), count(root, ".png")
        flag = ""
        if png < html:
            flag = f"   <- {html - png} to render"
            short.append(target)
        print(f"  {name:<{name_w}}  {target:<{target_w}}  html {html:>5}   png {png:>5}{flag}")

    print("\n  What the main study's notebooks open:\n")
    for _, rel_dst in EXPERIMENT_TREE[:2]:
        base = REPO_ROOT / rel_dst
        n = sum(1 for p in base.rglob("*.png") if not is_dead(p)) if base.is_dir() else 0
        flat = REPO_ROOT / rel_dst.rsplit("/metrics", 1)[0]
        n_base = len([p for p in flat.glob("*.png")]) if flat.is_dir() else 0
        state = "ready" if n and n_base else "EMPTY -- re-run without --report"
        print(f"    {rel_dst:<45}  {n:>5} scaled + {n_base:>3} baseline   {state}")

    print("\n  Counts exclude abandoned drafts (folders named '...ignore') and")
    print("  notebook checkpoints, which no condition reads.")

    if not short:
        print("\n  Every tree is fully rendered. Nothing to render.")
    else:
        print("\n  Render the short trees on a machine with Chromium:\n")
        print("    python3 utils/render_html_to_png.py --check-env")
        print("    python3 utils/render_html_to_png.py " + " ".join(short))

    print("\n  Then confirm nothing drifted:\n")
    print("    python3 reproducibility/verify_stimuli.py --manifest")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="count only, unpack nothing")
    ap.add_argument("--verify", action="store_true",
                    help="checksum every already-present baseline against the archive")
    args = ap.parse_args()

    rc = 0
    if not args.report:
        rc |= unpack(args.verify)
        rc |= install(args.verify)
    rc |= report()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
