"""
Paired A/B across two poster PROFILES (authority pilot).

The existing runners in `e1_optimized.py` cannot express this comparison:

  * `run_e1_baseline_paired` always takes the `_c` file from one directory and the `_i` file
    from the other, so it can pair Dr-incorrect against Remy-correct, but not Dr-correct
    against Remy-correct (both `_c`).
  * `run_e1_correct_vs_correct_paired` holds the variant constant but varies *engagement scale*
    within a single profile.

This module generalises the pairing: each side is an arbitrary (directory, variant) pair, so any
profile × variant combination can be pitted against any other.

Motivation — testing whether an authority cue outranks factual correctness. With profiles
`remy-ashford` (plain) and `dr-remy-ashford` ("Dr." + verified badge), the four cross-profile
pairings are:

  1. Dr+incorrect  vs  Remy+correct    authority vs correctness  <- the headline test
  2. Dr+correct    vs  Remy+correct    pure authority, correctness held equal
  3. Dr+incorrect  vs  Remy+incorrect  pure authority, both wrong
  4. Dr+correct    vs  Remy+incorrect  both cues agree; sanity check

Pairings 2 and 3 are the controls that make 1 interpretable: without them, a preference for the
Dr post in pairing 1 cannot be distinguished from a blanket preference for "Dr." irrespective of
what the post claims. This mirrors the role `run_e1_correct_vs_correct_paired` plays for the
engagement manipulation.

A/B ORDER: randomised per post with `random.Random(seed + int(num))` — the same formula
`run_e1_baseline_paired` uses, so a given post lands in the same slot across all four pairings
and across the main study's own baseline run. Differences between pairings are therefore not
attributable to a reshuffle.

POSITION BIAS: `analyse_profile_paired` reports the A-rate alongside every result, and refuses to
report a preference as meaningful when the model answers one slot on >=90% of trials. Several
models in this study default to a fixed slot and score ~50% mechanically (Section 5.1); that has
already caused findings to be misread more than once, so the check is built in rather than left
to the analyst.
"""
import json
import random
from pathlib import Path

SUFFIX = {"correct": "c", "incorrect": "i"}


def run_e1_profile_paired(selected_numbers, side_a, side_b, model, processor, device,
                          output_dir: Path, seed: int, prompt: str, output_filename: str,
                          inference_fn):
    """Pair two (profile, variant) sides against each other, one post number at a time.

    side_a / side_b: dicts with keys
        dir     : Path to the directory holding that side's PNGs
        variant : "correct" | "incorrect"
        label   : short name recorded in the output, e.g. "dr" / "plain"
    """
    output_path = output_dir / output_filename
    if output_path.exists():
        results = json.loads(output_path.read_text())
        done = {r["num"] for r in results}
        print(f"📋 Resuming — {len(done)} pairs already processed.")
    else:
        results, done = [], set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for num in selected_numbers:
        if num in done:
            continue
        pa = str(Path(side_a["dir"]) / f"{num}_remy_ashford_{SUFFIX[side_a['variant']]}.png")
        pb = str(Path(side_b["dir"]) / f"{num}_remy_ashford_{SUFFIX[side_b['variant']]}.png")

        # same formula as run_e1_baseline_paired -> identical slot assignment across pairings
        rng = random.Random(seed + int(num))
        if rng.random() < 0.5:
            (a_path, a_side), (b_path, b_side) = (pa, side_a), (pb, side_b)
        else:
            (a_path, a_side), (b_path, b_side) = (pb, side_b), (pa, side_a)

        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Post A:"},
            {"type": "image", "image": a_path},
            {"type": "text", "text": "Post B:"},
            {"type": "image", "image": b_path},
            {"type": "text", "text": prompt},
        ]}]
        answer = inference_fn(messages, model, processor, device).upper()
        chosen = a_side if answer == "A" else b_side if answer == "B" else None

        results.append({
            "num": num,
            "post_a_profile": a_side["label"], "post_a_variant": a_side["variant"],
            "post_b_profile": b_side["label"], "post_b_variant": b_side["variant"],
            "prompt": prompt,
            "answer": answer,
            "liked_profile": chosen["label"] if chosen else "invalid",
            "liked_variant": chosen["variant"] if chosen else "invalid",
        })
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"✅ {num} → {results[-1]['liked_profile']}/{results[-1]['liked_variant']} (answered {answer})")

    print(f"\n✅ Done. Results saved to {output_path}")
    return results


def analyse_profile_paired(output_dir: Path, output_filename: str, authority_label: str = "dr"):
    """Print the authority preference, the correctness preference, and the A-rate."""
    data = json.loads((Path(output_dir) / output_filename).read_text())
    valid = [r for r in data if r["answer"] in ("A", "B")]
    n, nv = len(data), len(valid)
    if not nv:
        print(f"{output_filename}: no valid answers ({n} trials)")
        return

    pct = lambda k: sum(1 for r in valid if r[k[0]] == k[1]) / nv * 100
    auth = pct(("liked_profile", authority_label))
    corr = pct(("liked_variant", "correct"))
    a_rate = sum(1 for r in valid if r["answer"] == "A") / nv * 100
    auth_in_a = sum(1 for r in valid if r["post_a_profile"] == authority_label) / nv * 100

    variants = {r["post_a_variant"] for r in valid} | {r["post_b_variant"] for r in valid}
    same_variant = len(variants) == 1

    print(f"\n=== {output_filename} ===")
    print(f"  n={n} valid={nv} invalid={n - nv}")
    print(f"  chose the '{authority_label}' post : {auth:5.1f}%   (it sat in slot A {auth_in_a:.1f}% of trials)")
    if not same_variant:
        print(f"  chose the CORRECT post      : {corr:5.1f}%")
    else:
        print(f"  (both sides {list(variants)[0]} — correctness is held constant, so any preference is pure profile)")
    print(f"  answered 'A'                : {a_rate:5.1f}%")

    if max(a_rate, 100 - a_rate) >= 90:
        print(f"  ⚠ POSITIONAL DEFAULT — the model answers one slot on {max(a_rate, 100-a_rate):.1f}% of trials.")
        print(f"    Any preference above is an artifact of where each post happened to land, not a choice.")
        print(f"    Do not report it as a profile or correctness effect.")
    elif max(a_rate, 100 - a_rate) >= 70:
        print(f"  ⚠ Marked positional lean ({max(a_rate, 100-a_rate):.1f}% one slot) — interpret with care.")
    return {"authority_pct": auth, "correct_pct": corr, "a_rate": a_rate, "n": n, "valid": nv}
