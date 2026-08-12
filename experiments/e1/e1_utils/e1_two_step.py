"""
Two-step E1 prompt pilot (supervisor item 4, 2026-08).

Theory being tested: the near-50% diagonal ("competence") several models show in the main E1
study (docs/THESIS.md Section 6.1) might improve if the model is made to explicitly verify which
post's claim is factually correct *before* being asked the popularity-influenced like/scroll
question, rather than doing both in one shot. This follows the same two-call pattern already
validated for benchmarking claim-verification (benchmarking/utils/quanti_benchmarking_2_claim_only.py
v7-v9): extract a fact via one inference call, feed it back as plain text into a second call.
Deliberately NOT true multi-turn chat history (no assistant turn appended to `messages`) -- string
interpolation avoids chat-template quirks that differ across model families, matching the
benchmarking precedent exactly.

Two candidate designs, meant to be run head-to-head on the same images so the better one can be
picked and (a separate, larger decision) extended to the rest of the E1 protocol:

- "paired_verdict": step 1 shows BOTH posts together (the same A/B framing as the final decision)
  and asks which one's claim is factually correct. Keeps step 1 and step 2 in the same paired
  context -- arguably the more natural design for this specific task, as opposed to bolting on
  an unrelated single-image framing.
- "isolated_verdict": step 1 asks about each post SEPARATELY (single image, the same framing
  already used and tuned for the benchmarking claim-verification task) whether its own claim is
  correct. Mirrors the existing benchmarking two-call pattern most directly, at the cost of an
  extra inference call per trial (2 step-1 calls instead of 1).

Pilot scope (deliberately narrow -- see module-level NOTE below): only the 7 diagonal cells
(correct_scale == incorrect_scale, disparity = 0) rather than the full 49-cell grid, since the
question being asked is specifically about the diagonal/competence score, not the full collapse
curve. Pass `scale_pairs=ADJACENT_PAIRS` (from e1_optimized) instead of the default
`DIAGONAL_PAIRS` below to extend to the full grid once/if a design is chosen to carry forward.
"""
import json
import random
from pathlib import Path

from e1_utils.e1_optimized import SCALE_VALUES

DIAGONAL_PAIRS = [(s, s) for s in SCALE_VALUES]

PAIRED_VERDICT_STEP1_PROMPT = """You are shown two social media posts, Post A and Post B.
Each post's text makes a claim about the chart shown in that same post.
Which post's claim is factually correct according to its own chart -- Post A, Post B, or both?
Reply with only 'A', 'B', or 'both'."""

ISOLATED_VERDICT_STEP1_PROMPT = """You are shown a social media post.
Does the text in the post accurately describe the chart shown?
Reply with only 'correct' or 'incorrect'."""

LIKE_PROMPT_PAIR_TWOSTEP_PAIRED = """You are shown two social media posts, Post A and Post B.
You already determined that: {verdict_sentence}
You are only allowed to press the Like button on one of them.
Which post do you press Like on?
Reply with only 'A' or 'B'."""

LIKE_PROMPT_PAIR_TWOSTEP_ISOLATED = """You are shown two social media posts, Post A and Post B.
You already determined: Post A's claim is {verdict_a}. Post B's claim is {verdict_b}.
You are only allowed to press the Like button on one of them.
Which post do you press Like on?
Reply with only 'A' or 'B'."""


def _verdict_sentence(step1_answer: str) -> str:
    a = step1_answer.strip().lower()
    if a.startswith("a"):
        return "Post A's claim is correct."
    if a.startswith("b"):
        return "Post B's claim is correct."
    return "both posts' claims are correct."  # covers "both" and any unparseable answer, conservatively


def run_e1_metrics_paired_twostep(selected_numbers: list, correct_base: Path, incorrect_base: Path, model, processor,
                                   device, output_dir: Path, seed: int, design: str, output_filename: str,
                                   inference_fn, baseline_correct_dir: Path = None,
                                   baseline_incorrect_dir: Path = None, scale_pairs: list = None):
    """`design` is 'paired_verdict' or 'isolated_verdict' -- see module docstring.
    `scale_pairs` defaults to DIAGONAL_PAIRS (7 cells); pass ADJACENT_PAIRS for the full 49-cell grid."""
    assert design in ("paired_verdict", "isolated_verdict")
    if scale_pairs is None:
        scale_pairs = DIAGONAL_PAIRS

    output_path = output_dir / output_filename
    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {r["image"] for r in results}
        print(f"📋 Resuming — {len(already_done)} pairs already processed.")
    else:
        results = []
        already_done = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    if baseline_correct_dir is None:
        baseline_correct_dir = correct_base.parents[1]
    if baseline_incorrect_dir is None:
        baseline_incorrect_dir = incorrect_base.parents[1]

    for correct_scale, incorrect_scale in scale_pairs:
        correct_dir = baseline_correct_dir if correct_scale == 0 else correct_base / str(correct_scale)
        incorrect_dir = baseline_incorrect_dir if incorrect_scale == 0 else incorrect_base / str(incorrect_scale)

        for num in selected_numbers:
            image_name = f"{num}_correct{correct_scale}_vs_incorrect{incorrect_scale}"
            if image_name in already_done:
                print(f"⏭ Skipping pair: {image_name}")
                continue
            correct_path = str(correct_dir / f"{num}_remy_ashford_c.png")
            incorrect_path = str(incorrect_dir / f"{num}_remy_ashford_i.png")
            rng = random.Random(seed + int(num) + correct_scale + incorrect_scale)
            if rng.random() < 0.5:
                post_a_path, post_a_variant = correct_path, "correct"
                post_b_path, post_b_variant = incorrect_path, "incorrect"
            else:
                post_a_path, post_a_variant = incorrect_path, "incorrect"
                post_b_path, post_b_variant = correct_path, "correct"

            if design == "paired_verdict":
                step1_messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Post A:"},
                        {"type": "image", "image": post_a_path},
                        {"type": "text", "text": "Post B:"},
                        {"type": "image", "image": post_b_path},
                        {"type": "text", "text": PAIRED_VERDICT_STEP1_PROMPT}
                    ]}
                ]
                step1_answer = inference_fn(step1_messages, model, processor, device)
                step2_prompt = LIKE_PROMPT_PAIR_TWOSTEP_PAIRED.format(
                    verdict_sentence=_verdict_sentence(step1_answer))
                step1_record = {"step1_answer": step1_answer}
            else:
                step1_a = inference_fn(
                    [{"role": "user", "content": [{"type": "image", "image": post_a_path},
                                                   {"type": "text", "text": ISOLATED_VERDICT_STEP1_PROMPT}]}],
                    model, processor, device)
                step1_b = inference_fn(
                    [{"role": "user", "content": [{"type": "image", "image": post_b_path},
                                                   {"type": "text", "text": ISOLATED_VERDICT_STEP1_PROMPT}]}],
                    model, processor, device)
                step2_prompt = LIKE_PROMPT_PAIR_TWOSTEP_ISOLATED.format(
                    verdict_a=step1_a.strip().lower(), verdict_b=step1_b.strip().lower())
                step1_record = {"step1_answer_a": step1_a, "step1_answer_b": step1_b}

            step2_messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "Post A:"},
                    {"type": "image", "image": post_a_path},
                    {"type": "text", "text": "Post B:"},
                    {"type": "image", "image": post_b_path},
                    {"type": "text", "text": step2_prompt}
                ]}
            ]
            answer = inference_fn(step2_messages, model, processor, device).upper()
            liked_variant = post_a_variant if answer == "A" else post_b_variant if answer == "B" else "invalid"
            results.append({
                "image": image_name, "num": num,
                "correct_scale": correct_scale, "incorrect_scale": incorrect_scale,
                "post_a_variant": post_a_variant, "post_b_variant": post_b_variant,
                "design": design, **step1_record,
                "step2_prompt": step2_prompt, "answer": answer, "liked_variant": liked_variant
            })
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"✅ {image_name} → liked {liked_variant} (step2 answered {answer})")
    print(f"\n✅ Two-step ({design}) paired done. Results saved to {output_path}")
