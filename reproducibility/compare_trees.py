#!/usr/bin/env python3
"""
Compare the stimulus images in benchmarking/ against the ones tracked under
spotify_pie_plot/, before consolidating on one of them.

Run this on the machine that has the real benchmarking/ tree (the GPU server),
before `git add`. It answers one question: are the images you actually ran the
experiments on byte-identical to the ones the repository ships?

    python3 reproducibility/compare_trees.py

If every file matches, adding the benchmarking tree to git creates no new blobs,
because git stores by content and those blobs are already in the repository under
the other path. The push is then small, and the spotify_pie_plot copies are safe
to remove afterwards.

If something differs, stop: the images the results came from are not the images
the repository ships, and which one is authoritative has to be settled first.

    --show N    list up to N differing files per group (default 10)
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONDITIONS = ["realistic", "likes_only", "likes_only_noise", "uniform"]
SCALES = ["10", "100", "1000", "10000", "100000", "1000000"]
VARIANTS = {"correct": "c", "incorrect": "i"}


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def compare_metrics(show: int):
    """benchmarking/<v>/remy-ashford/metrics/<cond>/<scale>/ vs the spotify tree."""
    print("Engagement-scaled images\n")
    total = same = diff = missing_b = missing_s = 0
    differing = []
    for variant in VARIANTS:
        bench = REPO_ROOT / f"benchmarking/{variant}/remy-ashford/metrics"
        spot = REPO_ROOT / f"spotify_pie_plot/pie_plot_posts/metrics/remy-ashford/{variant}/PNGs"
        for cond in CONDITIONS:
            for scale in SCALES:
                b_dir, s_dir = bench / cond / scale, spot / cond / scale
                names = set()
                if b_dir.is_dir():
                    names |= {p.name for p in b_dir.glob("*.png")}
                if s_dir.is_dir():
                    names |= {p.name for p in s_dir.glob("*.png")}
                for name in sorted(names):
                    total += 1
                    b, s = b_dir / name, s_dir / name
                    if not b.exists():
                        missing_b += 1
                    elif not s.exists():
                        missing_s += 1
                    elif md5(b.read_bytes()) == md5(s.read_bytes()):
                        same += 1
                    else:
                        diff += 1
                        differing.append(f"{variant}/{cond}/{scale}/{name}")
    print(f"  identical: {same}   differing: {diff}")
    print(f"  only in benchmarking: {missing_b}   only in spotify_pie_plot: {missing_s}")
    for path in differing[:show]:
        print(f"    DIFFERS  {path}")
    if len(differing) > show:
        print(f"    ... and {len(differing) - show} more")
    return diff, missing_b, missing_s


def compare_baselines(show: int):
    """benchmarking/<v>/remy-ashford/*.png vs the tracked baseline archives."""
    print("\nBaseline images\n")
    same = diff = missing = 0
    differing = []
    for variant, suffix in VARIANTS.items():
        bench = REPO_ROOT / f"benchmarking/{variant}/remy-ashford"
        archive = (REPO_ROOT / "spotify_pie_plot/pie_plot_posts/baselines"
                   / variant / "PNGs" / f"remy_ashford_{suffix}_pngs.zip")
        if not archive.exists():
            print(f"  [missing archive] {archive.relative_to(REPO_ROOT)}")
            continue
        with zipfile.ZipFile(archive) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".png")):
                local = bench / Path(name).name
                if not local.exists():
                    missing += 1
                elif md5(local.read_bytes()) == md5(zf.read(name)):
                    same += 1
                else:
                    diff += 1
                    differing.append(f"{variant}/{Path(name).name}")
    print(f"  identical: {same}   differing: {diff}   not on disk: {missing}")
    for path in differing[:show]:
        print(f"    DIFFERS  {path}")
    return diff, missing


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, default=10,
                    help="how many differing files to list per group")
    args = ap.parse_args()

    bench_root = REPO_ROOT / "benchmarking/correct/remy-ashford"
    if not any(bench_root.glob("*.png")):
        print("  benchmarking/correct/remy-ashford/ holds no PNGs on this machine.")
        print("  Run this where the real tree lives (the GPU server), not on a fresh clone.")
        return 1

    m_diff, only_b, only_s = compare_metrics(args.show)
    b_diff, b_missing = compare_baselines(args.show)

    print("\nVerdict\n")
    if m_diff or b_diff:
        print("  The two trees hold different images. Do not consolidate yet:")
        print("  settle which one produced the results before removing either.")
        return 1
    print("  Every image present in both trees is byte-identical.")
    if only_b:
        print(f"  {only_b} image(s) exist only in benchmarking/. Adding them to git writes")
        print("  new blobs, which is expected: the repository never had them.")
    if only_s:
        print(f"  {only_s} image(s) exist only under spotify_pie_plot/. Check these before")
        print("  removing that tree, they are not covered by the benchmarking copy.")
    if b_missing:
        print(f"  {b_missing} baseline(s) are in the archive but not loose on disk.")
    print("\n  Safe to track the benchmarking tree. Because git addresses by content,")
    print("  the matching files reuse blobs already in the repository, so the push")
    print("  carries only what is genuinely new.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
