import json
import re
import pandas as pd
from pathlib import Path
from IPython.display import display

GROUND_TRUTH = {
    "chart_percentages_pop": "23.5",
    "chart_percentages_latin": "11.0",
}

def normalize_number(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r'[,.](?=\d{3})', '', value)
    value = value.replace(',', '.')
    if '.' in value:
        try:
            value = str(float(value))
            value = re.sub(r'\.0$', '', value)
        except ValueError:
            pass
    return value

def run_accuracy_analysis(base_dir: Path, experiment_name: str, model_name: str):
    test_dir = base_dir / "outputs" / model_name / "quantitative" / experiment_name
    records = []
    for f in sorted(test_dir.glob("*.json")):
        data = json.loads(f.read_text())
        image_name = data["image"]
        variant = "correct" if image_name.endswith("_correct") else "incorrect"
        row = {"image": image_name, "variant": variant}
        for q, gt in GROUND_TRUTH.items():
            pred = normalize_number(data["answers"].get(q, ""))
            truth = normalize_number(gt)
            row[f"{q}_pred"] = pred
            row[f"{q}_true"] = truth
            row[f"{q}_correct"] = pred == truth
        records.append(row)
    df = pd.DataFrame(records)
    correct_cols = [f"{q}_correct" for q in GROUND_TRUTH]
    overall = pd.DataFrame({
        "metric": [
            "pop_accuracy_%",
            "latin_accuracy_%",
            "both_correct_%",
            "pop_accuracy_correct_variant_%",
            "pop_accuracy_incorrect_variant_%",
            "latin_accuracy_correct_variant_%",
            "latin_accuracy_incorrect_variant_%",
        ],
        "value": [
            round(df["chart_percentages_pop_correct"].mean() * 100, 2),
            round(df["chart_percentages_latin_correct"].mean() * 100, 2),
            round(df[correct_cols].all(axis=1).mean() * 100, 2),
            round(df[df["variant"] == "correct"]["chart_percentages_pop_correct"].mean() * 100, 2),
            round(df[df["variant"] == "incorrect"]["chart_percentages_pop_correct"].mean() * 100, 2),
            round(df[df["variant"] == "correct"]["chart_percentages_latin_correct"].mean() * 100, 2),
            round(df[df["variant"] == "incorrect"]["chart_percentages_latin_correct"].mean() * 100, 2),
        ]
    })
    print("=== Overall Summary ===")
    display(overall)
    print("=== Per Image Results ===")
    display(df)
    out_path = test_dir / f"accuracy_scores_{experiment_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")