import json
import pandas as pd
from pathlib import Path
from IPython.display import display

def extract_verdict(raw_answer: str) -> str:
    """Recovers a correct/incorrect verdict from a verbose response, not just a terse one --
    checks 'incorrect' before 'correct' since 'incorrect' contains 'correct' as a substring.
    Matches the lenient-matching precedent already used for COLOR_QUESTIONS in
    quanti_benchmarking_1_analysis.py."""
    text = raw_answer.strip().lower()
    if text in ("correct", "incorrect"):
        return text
    if "incorrect" in text:
        return "incorrect"
    if "correct" in text:
        return "correct"
    return text  # genuinely unparseable -- won't match either ground truth value
    

def run_accuracy_analysis(base_dir: Path, experiment_name: str, model_name: str):
    test_dir = base_dir / "outputs" / model_name / "quantitative" / experiment_name
    version_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    for version_dir in version_dirs:
        records = []
        for f in version_dir.glob("*.json"):
            data = json.loads(f.read_text())
            image_name = data["image"]

            ground_truth = "correct" if image_name.endswith("_correct") else "incorrect"
            prediction = extract_verdict(data["answers"]["post_claim_correct"])

            records.append({
                "image": image_name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": prediction == ground_truth
            })

        if not records:
            print(f"⚠️ No JSON files found in {version_dir.name}, skipping.")
            continue

        df = pd.DataFrame(records)
        prompt_text = json.loads(list(version_dir.glob("*.json"))[0].read_text()).get("prompt_text", "unknown")

        print(f"\n{'='*60}")
        print(f"Prompt Version : {version_dir.name}")
        print(f"Prompt Text    : {prompt_text}")
        print(f"{'='*60}")

        overall = pd.DataFrame({
            "metric": [
                "overall_accuracy_%",
                "accuracy_correct_posts_%",
                "accuracy_incorrect_posts_%"
            ],
            "value": [
                round(df["correct"].mean() * 100, 2),
                round(df[df["ground_truth"] == "correct"]["correct"].mean() * 100, 2),
                round(df[df["ground_truth"] == "incorrect"]["correct"].mean() * 100, 2)
            ]
        })
        print("=== Overall Summary ===")
        display(overall)

        print("=== Per Image Results ===")
        display(df)

        out_path = version_dir / f"accuracy_scores_{version_dir.name}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved to: {out_path}")