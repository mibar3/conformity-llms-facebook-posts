"""
Re-scores Gemma-E4B's test-2-gn-claim-only and test-4-metrics-realistic-claim-only
benchmarking results (original chart and simple-plot pilot) from the raw per-image JSON
files, fixing a scoring artifact in `benchmarking/utils/quanti_benchmarking_{2,4}_analysis.py`:
both scripts do `prediction == ground_truth` on the model's *entire* raw response
(`.strip().lower()`), so any response that isn't the bare word "correct"/"incorrect" is
scored wrong even when the model's stated verdict, buried in free-text reasoning, is right
(e.g. "...the claim matches the visualization.\n\ncorrect" was scored incorrect because the
whole string isn't "correct"). Not a caching bug and not new -- present in the original-chart
data since it was first written 2026-07-08/09, unrelated to the 2026-08-05 simple-plot rerun
(deterministic greedy decoding reproduced the same verbose text either way).

Fix: extract the LAST whole-word occurrence of "correct"/"incorrect" in the response (the
model's reasoning generally arrives at its stated verdict at the end) instead of requiring an
exact full-string match. Falls back to the old exact-match behaviour when the response already
is the bare word (the common case, left untouched).

Overwrites the affected accuracy_scores_v*.csv files in place (same schema, "prediction" and
"correct" columns updated) and prints a before/after accuracy table plus a rescue count so the
change is auditable. Pure stdlib -- see statistical_analysis/logprob_analysis.py for why.

Run: python3 statistical_analysis/rescore_claim_verdict.py
"""

import csv
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

TARGETS = [
    ("benchmarking/outputs/gemma-e4b", "test-2-gn-claim-only", "single"),
    ("benchmarking/outputs/gemma-e4b", "test-4-metrics-realistic-claim-only", "scale"),
    ("benchmarking_simple_plot/outputs/gemma-e4b", "test-2-gn-claim-only", "single"),
    ("benchmarking_simple_plot/outputs/gemma-e4b", "test-4-metrics-realistic-claim-only", "scale"),
]

VERDICT_RE = re.compile(r"\b(in)?correct\b")


def extract_verdict(raw_text):
    text = raw_text.strip().lower()
    if text in ("correct", "incorrect"):
        return text
    matches = VERDICT_RE.findall(text)
    if not matches:
        return None  # genuinely unparseable -- left as a miss, same as before
    return "incorrect" if matches[-1] == "in" else "correct"


def rescore_single(version_dir):
    """test-2 style: ground truth from filename suffix, one verdict per image."""
    rows = []
    for f in sorted(version_dir.glob("*.json")):
        data = json.loads(f.read_text())
        image_name = data["image"]
        ground_truth = "correct" if image_name.endswith("_correct") else "incorrect"
        raw = data["answers"]["post_claim_correct"]
        prediction = extract_verdict(raw)
        rows.append({
            "image": image_name,
            "ground_truth": ground_truth,
            "prediction": prediction if prediction is not None else raw.strip().lower(),
            "correct": prediction == ground_truth,
        })
    return rows


def rescore_scale(version_dir):
    """test-4 style: ground truth from filename, plus a scale_value field."""
    rows = []
    for f in sorted(version_dir.glob("*.json")):
        data = json.loads(f.read_text())
        image_name = data["image"]
        parts = image_name.split("_")
        variant = "correct" if "_correct_" in image_name else "incorrect"
        scale_value = int(parts[-1])
        raw = data["answers"]["post_claim_correct"]
        prediction = extract_verdict(raw)
        rows.append({
            "image": image_name,
            "scale_value": scale_value,
            "variant": variant,
            "ground_truth": variant,
            "prediction": prediction if prediction is not None else raw.strip().lower(),
            "correct": prediction == variant,
        })
    return rows


RESCORERS = {"single": rescore_single, "scale": rescore_scale}
FIELDNAMES = {
    "single": ["image", "ground_truth", "prediction", "correct"],
    "scale": ["image", "scale_value", "variant", "ground_truth", "prediction", "correct"],
}


def old_accuracy(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    k = sum(1 for r in rows if r["correct"].strip().lower() == "true")
    return n, k


def main():
    print(f"{'base':38s} {'test':38s} {'v':4s} {'n':>5s} {'old_acc':>8s} {'new_acc':>8s} {'rescued':>8s}")
    for base, test, mode in TARGETS:
        test_dir = ROOT_DIR / base / "quantitative" / test
        for version_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
            csv_path = version_dir / f"accuracy_scores_{version_dir.name}.csv"
            n_old, k_old = old_accuracy(csv_path)

            rows = RESCORERS[mode](version_dir)
            n_new = len(rows)
            k_new = sum(1 for r in rows if r["correct"])
            rescued = sum(1 for r in rows if r["correct"]) - k_old  # net change, not per-row rescue count

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES[mode])
                writer.writeheader()
                writer.writerows(rows)

            print(f"{base:38s} {test:38s} {version_dir.name:4s} {n_new:5d} "
                  f"{100*k_old/n_old:7.1f}% {100*k_new/n_new:7.1f}% {rescued:+8d}")


if __name__ == "__main__":
    main()
