"""
Mini chart-type benchmark, single script, gemma4-12b only. Loads the model ONCE and runs, per
chart type, (1) the behavioral like/scroll question the real pilot uses, and (2) a chart-reading
perception check adapted from the main study's Phase 1 perception validation
(benchmarking/utils/quanti_benchmarking_2_claim_only.py, prompt v3) -- "does the caption's claim
match the chart", scored against the known ground truth (which folder, correct/incorrect, each
image lives in), and (3) a stricter value-extraction check -- ask for the actual Year 10 solar and
wind numbers and compare against the fabricated ground truth those charts were built from
(generate_solar_wind_series). (2) can pass by guessing the right side of a binary question without
really reading the numbers; (3) can't. Together these separate three different explanations for
gemma4-12b's flat like/scroll behavior: it might not be able to read these charts at all (both
checks fail), it might get the gist but not precise values (claim-verify passes, extraction
doesn't), or it might read them fine and the flatness is purely about the like/scroll decision
itself (both checks pass even though like/scroll stays flat).

Loads the model once and covers all three chart-type variants in climate_pilot/chart_type_benchmark/
(bar_grouped, bar_single_year, area_overlay), then prints one side-by-side comparison table. This
replaces doing one more full notebook round-trip per hypothesis -- see
chart_type_benchmark/generate_benchmark_charts.py for why.

Run this on the GPU server (needs the model, not Chromium):
    python3 climate_pilot/chart_type_benchmark/run_benchmark_gemma12b.py

Writes per-type results to chart_type_benchmark/<chart_type>/outputs/, same file names as the
real pilot notebooks, so they can be inspected the same way if a chart type turns out to be worth
promoting into a real diagnostic.
"""
import json
import re
import sys
import statistics as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "experiments/e1"))
sys.path.insert(0, str(ROOT_DIR))
sys.path.append("/home/jovyan")

from climate_pilot.generate_climate_stimuli import generate_solar_wind_series  # noqa: E402

from config_hf_token import HF_TOKEN  # noqa: E402
from huggingface_hub import login  # noqa: E402
login(token=HF_TOKEN)

import torch  # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402

from e1_utils.e1_optimized import (  # noqa: E402
    LIKE_PROMPT_SINGLE, LIKE_CANDIDATES_SINGLE, run_e1_baseline, run_e1_baseline_logprobs,
)
from e1_utils.inference_gemma import run_inference_gemma, run_inference_with_scores_gemma  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
CHART_TYPES = ["bar_grouped", "bar_single_year", "area_overlay"]

# Chart-reading perception check -- adapted from the main study's Phase 1 prompt v3
# (benchmarking/utils/quanti_benchmarking_2_claim_only.py), same "reply with only 'correct' or
# 'incorrect'" scoring convention, just reworded for the solar/wind investment domain instead of
# Pop/Latin music-genre percentages.
CLAIM_VERIFY_PROMPT = (
    "The post text claims one energy source's investment ended up higher than the other's by "
    "Year 10. Look at the values in the chart to verify this claim. If the text correctly "
    "describes what the chart shows, reply 'correct'. If not, reply 'incorrect'."
)
CLAIM_VERIFY_CANDIDATES = ["correct", "incorrect"]

# Stricter perception check -- can it actually read the two numbers off the chart, not just guess
# the right side of a binary question? Scored against the fabricated ground truth
# (generate_solar_wind_series is deterministic per post number, same data used to build the chart).
EXTRACT_VALUES_PROMPT = (
    "Look only at the chart, ignore the post text above it. What is the Year 10 value for solar "
    "and the Year 10 value for wind? Reply with only the two numbers in the format "
    "'solar_value, wind_value', no units, no explanation."
)
EXTRACT_VALUE_TOLERANCE = 0.15  # relative error allowed to still count as "read correctly"

# Ground truth Year 10 values for posts 001-010, same fabricated data the charts were built from.
GROUND_TRUTH_YEAR10 = {}
for _num in range(1, 11):
    _solar, _wind, _ = generate_solar_wind_series(_num)
    GROUND_TRUTH_YEAR10[f"{_num:03d}"] = (_solar[-1], _wind[-1])

MODEL_ID = "google/gemma-4-12B-it"
print(f"Loading {MODEL_ID} ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")
device = model.device
print("Model loaded.\n")


def load_images(chart_type: str):
    correct_dir = BENCH_DIR / chart_type / "posts" / "correct" / "PNGs"
    incorrect_dir = BENCH_DIR / chart_type / "posts" / "incorrect" / "PNGs"
    images = []
    for p in sorted(correct_dir.glob("*_remy_ashford_c.png")):
        num = p.name.split("_")[0]
        images.append((f"{num}_correct", str(p)))
        images.append((f"{num}_incorrect", str(incorrect_dir / f"{num}_remy_ashford_i.png")))
    return images


results_summary = []

for chart_type in CHART_TYPES:
    print(f"\n{'=' * 60}\n{chart_type}\n{'=' * 60}")
    images = load_images(chart_type)
    if not images:
        print(f"  [SKIP] No rendered PNGs found for {chart_type} -- run render_benchmark_charts.py first.")
        continue

    output_dir = BENCH_DIR / chart_type / "outputs"

    # --- Behavioral question: like/scroll, same as the real pilot ---
    run_e1_baseline(images, model, processor, device, output_dir,
                     prompt=LIKE_PROMPT_SINGLE, output_filename="e1_results_baseline.json",
                     inference_fn=run_inference_gemma)
    run_e1_baseline_logprobs(images, model, processor, device, output_dir,
                              prompt=LIKE_PROMPT_SINGLE, candidates=LIKE_CANDIDATES_SINGLE,
                              output_filename="e1_results_baseline_logprobs.json",
                              score_fn=run_inference_with_scores_gemma)

    data = json.loads((output_dir / "e1_results_baseline_logprobs.json").read_text())
    like_correct = [d["candidates"]["like"]["prob_forced_choice"] for d in data if d["variant"] == "correct"]
    like_incorrect = [d["candidates"]["like"]["prob_forced_choice"] for d in data if d["variant"] == "incorrect"]
    answers = json.loads((output_dir / "e1_results_baseline.json").read_text())
    unique_answers = sorted(set(a["answer"] for a in answers))

    # --- Perception check: can it read the chart at all, independent of like/scroll? ---
    run_e1_baseline(images, model, processor, device, output_dir,
                     prompt=CLAIM_VERIFY_PROMPT, output_filename="e1_results_claim_verify.json",
                     inference_fn=run_inference_gemma)
    run_e1_baseline_logprobs(images, model, processor, device, output_dir,
                              prompt=CLAIM_VERIFY_PROMPT, candidates=CLAIM_VERIFY_CANDIDATES,
                              output_filename="e1_results_claim_verify_logprobs.json",
                              score_fn=run_inference_with_scores_gemma)

    claim_answers = json.loads((output_dir / "e1_results_claim_verify.json").read_text())
    n_correct = sum(1 for d in claim_answers if d["answer"] == d["variant"])
    claim_accuracy = n_correct / len(claim_answers)
    claim_unique_answers = sorted(set(d["answer"] for d in claim_answers))

    # --- Stricter perception check: can it read the actual numbers? ---
    run_e1_baseline(images, model, processor, device, output_dir,
                     prompt=EXTRACT_VALUES_PROMPT, output_filename="e1_results_extract_values.json",
                     inference_fn=run_inference_gemma)

    extract_answers = json.loads((output_dir / "e1_results_extract_values.json").read_text())
    n_parsed, n_within_tolerance = 0, 0
    for d in extract_answers:
        num = d["image"].split("_")[0]
        true_solar, true_wind = GROUND_TRUTH_YEAR10[num]
        nums = re.findall(r"-?\d+\.?\d*", d["answer"])
        if len(nums) < 2:
            continue
        n_parsed += 1
        guess_solar, guess_wind = float(nums[0]), float(nums[1])
        solar_err = abs(guess_solar - true_solar) / true_solar
        wind_err = abs(guess_wind - true_wind) / true_wind
        if solar_err <= EXTRACT_VALUE_TOLERANCE and wind_err <= EXTRACT_VALUE_TOLERANCE:
            n_within_tolerance += 1
    extract_parse_rate = n_parsed / len(extract_answers)
    extract_accuracy = n_within_tolerance / len(extract_answers)

    results_summary.append({
        "chart_type": chart_type,
        "unique_answers": unique_answers,
        "like_correct_mean": st.mean(like_correct), "like_correct_stdev": st.pstdev(like_correct),
        "like_incorrect_mean": st.mean(like_incorrect), "like_incorrect_stdev": st.pstdev(like_incorrect),
        "claim_unique_answers": claim_unique_answers,
        "claim_accuracy": claim_accuracy,
        "extract_parse_rate": extract_parse_rate,
        "extract_accuracy": extract_accuracy,
    })

print(f"\n\n{'=' * 120}\nBENCHMARK SUMMARY -- gemma4-12b, baseline\n{'=' * 120}")
header = (f"{'chart_type':<18}{'like answers':<15}{'P(like) correct':<20}{'P(like) incorrect':<20}"
          f"{'claim answers':<18}{'claim acc':<12}{'extract parsed':<16}{'extract acc':<12}")
print(header)
for r in results_summary:
    like_correct_str = f"{r['like_correct_mean']:.4f}±{r['like_correct_stdev']:.4f}"
    like_incorrect_str = f"{r['like_incorrect_mean']:.4f}±{r['like_incorrect_stdev']:.4f}"
    claim_acc_str = f"{r['claim_accuracy'] * 100:.1f}%"
    extract_parsed_str = f"{r['extract_parse_rate'] * 100:.1f}%"
    extract_acc_str = f"{r['extract_accuracy'] * 100:.1f}%"
    print(f"{r['chart_type']:<18}{str(r['unique_answers']):<15}{like_correct_str:<20}"
          f"{like_incorrect_str:<20}{str(r['claim_unique_answers']):<18}{claim_acc_str:<12}"
          f"{extract_parsed_str:<16}{extract_acc_str:<12}")

print("\nHow to read this:")
print("- 'like answers' with only one unique value = still flat (matches line/scatter results).")
print("- 'claim acc' near 50% = can't distinguish correct/incorrect captions at all. Near 100% =")
print("  it gets the gist right, but that's gameable by other cues, not proof it read the numbers.")
print("- 'extract acc' (values within 15% of ground truth for BOTH solar and wind) is the strict")
print("  test: near 0% despite high claim accuracy = it's pattern-matching the gist, not really")
print("  reading the chart. High extract acc + still-flat like/scroll = the flatness is purely a")
print("  decision/framing effect, not perception, and this chart type isn't the fix either.")
print("- A chart type is worth promoting to a real diagnostic only if 'like answers' shows >1")
print("  unique value AND a real gap between P(like) correct vs incorrect (not just nonzero")
print("  stdev, which can still be noise around a saturated prior).")
