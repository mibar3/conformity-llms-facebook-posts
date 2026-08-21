"""
Authority x engagement grid: "Dr. Remy Ashford" vs plain "Remy Ashford", both posts CORRECT,
engagement varied independently on each side.

WHY THIS EXISTS
---------------
Neither existing runner can express the comparison:

  * `run_e1_correct_vs_correct_paired` varies engagement scale but stays inside ONE profile, so
    its diagonal (scale_a == scale_b) is degenerate -- the two images are then byte-identical
    apart from slot, and `correct_vs_correct_analysis.py` discards those cells
    (`rows = [r for r in records if r["scale_a"] != r["scale_b"]]`, n=4200 not 4900).
  * `run_e1_profile_paired` (e1_profile.py) varies profile but holds engagement at zero.

Putting "Dr." on one side revives the diagonal: equal engagement, equal claim, one difference.
The diagonal becomes the pure-authority test measured at seven engagement levels, and Stage 1's
"pure_authority_both_correct" pairing is exactly its (0, 0) cell.

WHAT THE THREE REGIONS ANSWER
-----------------------------
  diagonal  (dr == plain)  7 cells   does the authority preference survive as popularity rises?
  conflict  (dr <  plain)  21 cells  authority vs popularity -- how many likes outweigh a "Dr."?
  aligned   (dr >  plain)  21 cells  both cues agree: saturation, and whether the effect is
                                     symmetric or authority is special

The matching no-authority control already exists: the off-diagonal cells of
`e1_results_metrics_correct_vs_correct_paired.json` are the same engagement contrasts between two
plain posts. Subtracting the two grids cell by cell isolates the authority effect, and whether
that difference grows or shrinks with the engagement gap distinguishes Social Impact Theory
(multiplicative: strength x number) from heuristic-sufficiency substitution.

CELL ORDER
----------
Cells run diagonal -> conflict -> aligned, most informative first, because the aligned triangle is
the region most likely to sit at ceiling (models already follow engagement 92-99% in
correct_vs_correct and the "Dr." post 98-100% at zero engagement). Combined with the resume logic
inherited from the other runners, the run can be stopped at any point and still yield a complete,
analysable design rather than a partial grid.

A/B ORDER
---------
`random.Random(seed + int(num) + dr_scale + plain_scale)` -- the same shape as
`run_e1_correct_vs_correct_paired`. At (0, 0) it reduces to `seed + int(num)`, which is the formula
`run_e1_profile_paired` and `run_e1_baseline_paired` use, so the diagonal's zero cell reproduces
Stage 1's slot assignment exactly and the two runs are directly comparable.

TIES
----
`liked_higher_engagement` is None on the diagonal. It is not False and not True: with engagement
equal there is no higher side. Writing True there is what made the tied cells trivially "correct"
in the original correct-vs-correct output and is why those cells had to be filtered out downstream.
"""
import json
import random
from pathlib import Path

SCALE_VALUES = [0, 10, 100, 1000, 10000, 100000, 1000000]


def build_grid_cells(scales=None):
    """(dr_scale, plain_scale, region) for the full grid, most informative region first."""
    scales = scales or SCALE_VALUES
    diagonal = [(s, s, "diagonal") for s in scales]
    conflict = [(d, p, "conflict") for d in scales for p in scales if d < p]
    aligned = [(d, p, "aligned") for d in scales for p in scales if d > p]
    return diagonal + conflict + aligned


def _dir_for(scale, metrics_base: Path, baseline_dir: Path) -> Path:
    """Scale 0 lives in the profile's baseline directory, every other scale under metrics/."""
    return Path(baseline_dir) if scale == 0 else Path(metrics_base) / str(scale)


def run_e1_profile_grid_paired(selected_numbers, dr_metrics_base: Path, plain_metrics_base: Path,
                               dr_baseline_dir: Path, plain_baseline_dir: Path,
                               model, processor, device, output_dir: Path, seed: int, prompt: str,
                               output_filename: str, inference_fn, cells=None):
    """Run the authority x engagement grid. Both sides are the CORRECT variant throughout.

    dr_metrics_base    : .../benchmarking/correct/dr-remy-ashford/metrics/realistic
    plain_metrics_base : .../benchmarking/correct/remy-ashford/metrics/realistic
    dr_baseline_dir    : .../benchmarking/correct/dr-remy-ashford        (scale 0)
    plain_baseline_dir : .../benchmarking/correct/remy-ashford          (scale 0)
    """
    cells = cells or build_grid_cells()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {r["image"] for r in results}
        print(f"📋 Resuming — {len(already_done)} pairs already processed.")
    else:
        results, already_done = [], set()

    total = len(cells) * len(selected_numbers)
    print(f"Grid: {len(cells)} cells x {len(selected_numbers)} posts = {total} trials")

    for dr_scale, plain_scale, region in cells:
        dir_dr = _dir_for(dr_scale, dr_metrics_base, dr_baseline_dir)
        dir_plain = _dir_for(plain_scale, plain_metrics_base, plain_baseline_dir)

        for num in selected_numbers:
            image_name = f"{num}_dr{dr_scale}_vs_plain{plain_scale}"
            if image_name in already_done:
                continue

            path_dr = str(dir_dr / f"{num}_remy_ashford_c.png")
            path_plain = str(dir_plain / f"{num}_remy_ashford_c.png")

            # reduces to seed + int(num) at (0, 0) -> same slots as the Stage 1 authority run
            rng = random.Random(seed + int(num) + dr_scale + plain_scale)
            if rng.random() < 0.5:
                (a_path, a_prof, a_scale) = (path_dr, "dr", dr_scale)
                (b_path, b_prof, b_scale) = (path_plain, "plain", plain_scale)
            else:
                (a_path, a_prof, a_scale) = (path_plain, "plain", plain_scale)
                (b_path, b_prof, b_scale) = (path_dr, "dr", dr_scale)

            messages = [{"role": "user", "content": [
                {"type": "text", "text": "Post A:"},
                {"type": "image", "image": a_path},
                {"type": "text", "text": "Post B:"},
                {"type": "image", "image": b_path},
                {"type": "text", "text": prompt},
            ]}]
            answer = inference_fn(messages, model, processor, device).upper()

            if answer == "A":
                liked_profile, liked_scale = a_prof, a_scale
            elif answer == "B":
                liked_profile, liked_scale = b_prof, b_scale
            else:
                liked_profile, liked_scale = "invalid", "invalid"

            if liked_profile == "invalid" or dr_scale == plain_scale:
                liked_higher = None          # no higher side on the diagonal -- see module docstring
            else:
                liked_higher = liked_scale == max(dr_scale, plain_scale)

            results.append({
                "image": image_name, "num": num, "region": region,
                "dr_scale": dr_scale, "plain_scale": plain_scale,
                "post_a_profile": a_prof, "post_a_scale": a_scale,
                "post_b_profile": b_prof, "post_b_scale": b_scale,
                "prompt": prompt, "answer": answer,
                "liked_profile": liked_profile, "liked_scale": liked_scale,
                "liked_dr": None if liked_profile == "invalid" else liked_profile == "dr",
                "liked_higher_engagement": liked_higher,
            })
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"✅ {image_name} [{region}] → {liked_profile} (answered {answer})")

    print(f"\n✅ Done. Results saved to {output_path}")
    return results


def analyse_profile_grid(output_dir: Path, output_filename: str):
    """Per-region summary, plus the position cross-check that the aggregate alone cannot give."""
    data = json.loads((Path(output_dir) / output_filename).read_text())
    valid = [r for r in data if r["answer"] in ("A", "B")]
    if not valid:
        print(f"{output_filename}: no valid answers ({len(data)} trials)")
        return

    print(f"\n=== {output_filename} ===")
    print(f"  n={len(data)} valid={len(valid)} invalid={len(data) - len(valid)}")

    summary = {}
    for region in ("diagonal", "conflict", "aligned"):
        sub = [r for r in valid if r["region"] == region]
        if not sub:
            continue
        dr = sum(1 for r in sub if r["liked_dr"]) / len(sub) * 100
        a_rate = sum(1 for r in sub if r["answer"] == "A") / len(sub) * 100
        dr_in_a = sum(1 for r in sub if r["post_a_profile"] == "dr") / len(sub) * 100
        summary[region] = {"n": len(sub), "pct_dr": dr, "a_rate": a_rate}

        print(f"\n  [{region}]  n={len(sub)}")
        print(f"    chose the 'Dr.' post : {dr:5.1f}%   (it sat in slot A {dr_in_a:.1f}% of trials)")
        print(f"    answered 'A'         : {a_rate:5.1f}%")

        # cross profile against position -- a preference that only holds in one slot is positional
        for slot in ("A", "B"):
            half = [r for r in sub if (r["post_a_profile"] == "dr") == (slot == "A")]
            if half:
                p = sum(1 for r in half if r["liked_dr"]) / len(half) * 100
                print(f"      'Dr.' in slot {slot} ({len(half):4d} trials) → chose it {p:5.1f}%")

        if max(a_rate, 100 - a_rate) >= 90:
            print(f"    ⚠ POSITIONAL DEFAULT ({max(a_rate, 100 - a_rate):.1f}% one slot) — not a preference.")

    # the exchange rate: where popularity overtakes the "Dr." post along the dr_scale == 0 row
    row = sorted({r["plain_scale"] for r in valid if r["region"] in ("diagonal", "conflict")
                  and r["dr_scale"] == 0})
    if row:
        print(f"\n  [exchange rate]  Dr. at 0 engagement vs plain post at:")
        for ps in row:
            sub = [r for r in valid if r["dr_scale"] == 0 and r["plain_scale"] == ps]
            if sub:
                p = sum(1 for r in sub if r["liked_dr"]) / len(sub) * 100
                flag = "  ← crossover" if p < 50 else ""
                print(f"    {ps:>8,} likes → chose 'Dr.' {p:5.1f}%  (n={len(sub)}){flag}")
    return summary
