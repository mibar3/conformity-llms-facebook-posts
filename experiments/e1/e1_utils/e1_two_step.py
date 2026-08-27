"""
Two-step E1 prompt pilot 
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

ISOLATED_VERDICT_STEP1_PROMPT = """Does the text in the post accurately describe the chart? Reply with only 'correct' or 'incorrect'."""

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
   """`design` is 'paired_verdict' or 'isolated_verdict' -- see module docstring. `isolated_verdict`
    uses the v1 wording validated by a cross-model comparison (docs/THESIS.md Section 5.3.2) -- an
    earlier v4-wording variant was tried and dropped 2026-08-27, kept only as static historical
    output files (e1_results_metrics_diagonal_twostep_isolated_verdict_v4.json), not as a runnable
    design. `scale_pairs` defaults to DIAGONAL_PAIRS (7 cells); pass ADJACENT_PAIRS for the full
    49-cell grid."""
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
                step1_prompt = ISOLATED_VERDICT_STEP1_PROMPT
                step1_a = inference_fn(
                    [{"role": "user", "content": [{"type": "image", "image": post_a_path},
                                                   {"type": "text", "text": step1_prompt}]}],
                    model, processor, device)
                step1_b = inference_fn(
                    [{"role": "user", "content": [{"type": "image", "image": post_b_path},
                                                   {"type": "text", "text": step1_prompt}]}],
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
