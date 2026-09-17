#!/usr/bin/env python3
"""
Prove that stimulus images are the ones the experiments actually ran on,
instead of trusting filenames. Filenames were reused across several generator
versions, so a name match means nothing here; pixels are the evidence.

Two independent checks:

--survivors   Full-image comparison of the baseline PNGs now in place -- unpacked
              from the tracked archives by reproducibility/prepare_stimuli.py, or
              regenerated -- against the ten original baseline images that also sit
              loose in the tree (eight Jupyter checkpoints, two loose copies). Same
              condition, so this compares the whole file including the all-zero
              engagement row. A match proves the set in place IS the original set.

--baselines   Structural comparison of every baseline PNG against the metrics
              stimuli for the same image number. Above the engagement bar, all 24
              metrics files for an image (4 conditions x 6 scales) are pixel-
              identical, so that region is a fingerprint of the profile, claim text
              and chart. Works for all 100 images, not just the ten survivors.
              The bar sits at a different height per image (roughly y=705 to 859,
              because chart label extents vary), so the boundary is found per image
              by diffing two scales rather than assumed.

--manifest    Compare every stimulus PNG against reproducibility/stimuli_manifest_before.json,
              captured at tag pre-path-refactor. Used to show the path refactor left
              every image byte-identical.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops
except ImportError:
    sys.exit("Pillow required:  python3 -m pip install --user Pillow")

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = REPO_ROOT / "spotify_pie_plot/pie_plot_posts/metrics/remy-ashford"
BASELINES = REPO_ROOT / "spotify_pie_plot/pie_plot_posts/baselines"
MANIFEST = REPO_ROOT / "reproducibility/stimuli_manifest_before.json"

SURVIVORS = [
    ("benchmarking/correct/remy-ashford/.ipynb_checkpoints/001_remy_ashford_c-checkpoint.png",   "001", "correct"),
    ("benchmarking/correct/remy-ashford/.ipynb_checkpoints/003_remy_ashford_c-checkpoint.png",   "003", "correct"),
    ("benchmarking/correct/remy-ashford/.ipynb_checkpoints/006_remy_ashford_c-checkpoint.png",   "006", "correct"),
    ("benchmarking/correct/remy-ashford/.ipynb_checkpoints/022_remy_ashford_c-checkpoint.png",   "022", "correct"),
    ("benchmarking/correct/remy-ashford/.ipynb_checkpoints/028_remy_ashford_c-checkpoint.png",   "028", "correct"),
    ("benchmarking/incorrect/remy-ashford/.ipynb_checkpoints/001_remy_ashford_i-checkpoint.png", "001", "incorrect"),
    ("benchmarking/incorrect/remy-ashford/.ipynb_checkpoints/012_remy_ashford_i-checkpoint.png", "012", "incorrect"),
    ("benchmarking/incorrect/remy-ashford/.ipynb_checkpoints/022_remy_ashford_i-checkpoint.png", "022", "incorrect"),
    ("benchmarking/correct/001_remy_ashford_c.png",                                              "001", "correct"),
    ("benchmarking/incorrect/001_remy_ashford_i.png",                                            "001", "incorrect"),
]
SUFFIX = {"correct": "c", "incorrect": "i"}


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def engagement_boundary(num: str, variant: str):
    """Top of the engagement region for this image, found by diffing two scales."""
    suf = SUFFIX[variant]
    a = METRICS / variant / "PNGs/realistic/10" / f"{num}_remy_ashford_{suf}.png"
    b = METRICS / variant / "PNGs/realistic/1000000" / f"{num}_remy_ashford_{suf}.png"
    if not (a.exists() and b.exists()):
        return None, None
    bbox = ImageChops.difference(Image.open(a).convert("RGB"),
                                 Image.open(b).convert("RGB")).getbbox()
    return (bbox[1] if bbox else None), a


def top_hash(path: Path, boundary: int) -> str:
    im = Image.open(path).convert("RGB")
    return md5(im.crop((0, 0, im.size[0], boundary)).tobytes())


def check_survivors() -> int:
    print("Full-image comparison: regenerated baselines vs surviving originals\n")
    missing = same = differ = 0
    for rel, num, variant in SURVIVORS:
        original = REPO_ROOT / rel
        regenerated = BASELINES / variant / "PNGs" / f"{num}_remy_ashford_{SUFFIX[variant]}.png"
        if not original.exists():
            print(f"  [skip] original gone:    {rel}"); missing += 1; continue
        if not regenerated.exists():
            print(f"  [skip] not regenerated:  {regenerated.relative_to(REPO_ROOT)}"); missing += 1; continue
        a, b = md5(original.read_bytes()), md5(regenerated.read_bytes())
        if a == b:
            print(f"  IDENTICAL  {num}_{variant}"); same += 1
        else:
            print(f"  DIFFERS    {num}_{variant}   original {a[:12]}  regenerated {b[:12]}"); differ += 1
    print(f"\n  identical: {same}   differs: {differ}   unavailable: {missing}")
    if differ == 0 and same:
        print("\n  The baseline PNGs in place are the originals the study ran on.")
    elif differ:
        print("\n  Renderer drift. Run --baselines to see whether the difference is above")
        print("  the engagement bar (chart/fonts) or only in the all-zero engagement row.")
    return 1 if differ else 0


def check_baselines() -> int:
    print("Structural comparison: every baseline PNG vs the metrics stimuli\n")
    ok = bad = miss = 0
    for variant in ("correct", "incorrect"):
        for n in range(1, 101):
            num = f"{n:03d}"
            png = BASELINES / variant / "PNGs" / f"{num}_remy_ashford_{SUFFIX[variant]}.png"
            if not png.exists():
                miss += 1; continue
            boundary, ref = engagement_boundary(num, variant)
            if boundary is None:
                miss += 1; continue
            if top_hash(png, boundary) == top_hash(ref, boundary):
                ok += 1
            else:
                print(f"  MISMATCH  {num}_{variant}  (above y={boundary})"); bad += 1
    print(f"\n  matching: {ok}   mismatched: {bad}   missing: {miss}")
    return 1 if bad else 0


def check_manifest() -> int:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    before = json.loads(MANIFEST.read_text())
    changed = gone = 0
    for rel, expected in sorted(before.items()):
        p = REPO_ROOT / rel
        if not p.exists():
            print(f"  GONE     {rel}"); gone += 1
        elif md5(p.read_bytes()) != expected:
            print(f"  CHANGED  {rel}"); changed += 1
    print(f"\n  checked: {len(before)}   changed: {changed}   moved or gone: {gone}")
    if changed == 0:
        print("  No stimulus image changed content.")
    return 1 if changed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survivors", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()
    if not any((args.survivors, args.baselines, args.manifest)):
        ap.print_help(); return 0
    rc = 0
    if args.survivors: rc |= check_survivors()
    if args.baselines: rc |= check_baselines()
    if args.manifest:  rc |= check_manifest()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
