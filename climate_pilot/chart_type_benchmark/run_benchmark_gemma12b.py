"""
Mini chart-type benchmark, single script, gemma4-12b only. Loads the model ONCE and runs
baseline like/scroll + logprobs across all three chart-type variants in
climate_pilot/chart_type_benchmark/ (bar_grouped, bar_single_year, area_overlay), then prints one
side-by-side comparison table. This replaces doing one more full notebook round-trip per
hypothesis -- see chart_type_benchmark/generate_benchmark_charts.py for why.

Run this on the GPU server (needs the model, not Chromium):
    python3 climate_pilot/chart_type_benchmark/run_benchmark_gemma12b.py

Writes per-type results to chart_type_benchmark/<chart_type>/outputs/, same file names as the
real pilot notebooks, so they can be inspected the same way if a chart type turns out to be worth
promoting into a real diagnostic.
"""
import sys
import statistics as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "experiments/e1"))
sys.path.append("/home/jovyan")

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

    run_e1_baseline(images, model, processor, device, output_dir,
                     prompt=LIKE_PROMPT_SINGLE, output_filename="e1_results_baseline.json",
                     inference_fn=run_inference_gemma)
    run_e1_baseline_logprobs(images, model, processor, device, output_dir,
                              prompt=LIKE_PROMPT_SINGLE, candidates=LIKE_CANDIDATES_SINGLE,
                              output_filename="e1_results_baseline_logprobs.json",
                              score_fn=run_inference_with_scores_gemma)

    import json
    data = json.loads((output_dir / "e1_results_baseline_logprobs.json").read_text())
    correct = [d["candidates"]["like"]["prob_forced_choice"] for d in data if d["variant"] == "correct"]
    incorrect = [d["candidates"]["like"]["prob_forced_choice"] for d in data if d["variant"] == "incorrect"]
    answers = json.loads((output_dir / "e1_results_baseline.json").read_text())
    unique_answers = sorted(set(a["answer"] for a in answers))

    results_summary.append({
        "chart_type": chart_type,
        "unique_answers": unique_answers,
        "correct_mean": st.mean(correct), "correct_stdev": st.pstdev(correct),
        "incorrect_mean": st.mean(incorrect), "incorrect_stdev": st.pstdev(incorrect),
    })

print(f"\n\n{'=' * 90}\nBENCHMARK SUMMARY -- P(like), baseline, gemma4-12b\n{'=' * 90}")
print(f"{'chart_type':<18}{'unique answers':<18}{'correct mean±sd':<22}{'incorrect mean±sd':<22}")
for r in results_summary:
    correct_str = f"{r['correct_mean']:.4f}±{r['correct_stdev']:.4f}"
    incorrect_str = f"{r['incorrect_mean']:.4f}±{r['incorrect_stdev']:.4f}"
    print(f"{r['chart_type']:<18}{str(r['unique_answers']):<18}{correct_str:<22}{incorrect_str:<22}")
print("\nA chart type is worth promoting to a real diagnostic if it shows >1 unique answer AND a")
print("real correct/incorrect gap in mean P(like) -- not just nonzero stdev (that can still be")
print("noise around a saturated prior, same as the line/scatter results already ruled out).")
