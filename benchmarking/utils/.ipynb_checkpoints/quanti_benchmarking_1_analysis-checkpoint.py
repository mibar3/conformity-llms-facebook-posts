import json
import re
import pandas as pd
from pathlib import Path
from IPython.display import display

NUMERIC_QUESTIONS = [
    "likes_count", "comment_count", "shares_count",
    "heart_reactions", "haha_reactions", "wow_reactions",
    "sad_reactions", "angry_reactions",
    "chart_percentages_pop", "chart_percentages_latin"
]

TEXT_QUESTIONS = [
    "profile_name", "profile_verification",
    "chart_color_pop", "chart_color_latin",
    "chart_largest_slice", "post_claim_correct"
]

ALL_QUESTIONS = NUMERIC_QUESTIONS + TEXT_QUESTIONS


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


def run_accuracy_analysis(base_dir: Path, experiment_name: str):
    df_truth = pd.read_csv(base_dir / "ground_truth/ground_truth_all.csv")

    experiment_dir = base_dir / "outputs/quantitative" / experiment_name
    version_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir()])

    for version_dir in version_dirs:
        records = []
        for f in version_dir.glob("*.json"):
            data = json.loads(f.read_text())
            row = {
                "image": data["image"],
                "prompt_version": data.get("prompt_version", version_dir.name),
                "prompt_text": data.get("prompt_text", "unknown")
            } | data["answers"]
            records.append(row)

        if not records:
            print(f"⚠️ No JSON files found in {version_dir.name}, skipping.")
            continue

        df_responses = pd.DataFrame(records)
        merged = df_responses.merge(df_truth, on="image", suffixes=("_pred", "_true"))

        for q in NUMERIC_QUESTIONS:
            pred_col, true_col = f"{q}_pred", f"{q}_true"
            if pred_col in merged.columns and true_col in merged.columns:
                merged[pred_col] = merged[pred_col].apply(normalize_number)
                merged[true_col] = merged[true_col].apply(normalize_number)

        for q in TEXT_QUESTIONS:
            pred_col, true_col = f"{q}_pred", f"{q}_true"
            if pred_col in merged.columns and true_col in merged.columns:
                merged[pred_col] = merged[pred_col].astype(str).str.strip().str.lower()
                merged[true_col] = merged[true_col].astype(str).str.strip().str.lower()

        for q in ALL_QUESTIONS:
            pred_col, true_col = f"{q}_pred", f"{q}_true"
            if pred_col in merged.columns and true_col in merged.columns:
                merged[f"{q}_correct"] = merged[pred_col] == merged[true_col]

        correct_cols = [f"{q}_correct" for q in ALL_QUESTIONS if f"{q}_correct" in merged.columns]

        prompt_text = df_responses["prompt_text"].iloc[0]
        print(f"\n{'='*60}")
        print(f"Prompt Version : {version_dir.name}")
        print(f"Prompt Text    : {prompt_text}")
        print(f"{'='*60}")

        accuracy = (
            merged[correct_cols].mean()
            .rename(index=lambda x: x.replace("_correct", ""))
            .mul(100).round(2).reset_index()
        )
        accuracy.columns = ["question", "accuracy_%"]
        print("=== Accuracy per question ===")
        display(accuracy)

        merged["all_correct"] = merged[correct_cols].all(axis=1)
        per_image = merged[["image", "all_correct"] + correct_cols].copy()
        per_image.columns = [c.replace("_correct", "") for c in per_image.columns]
        print("=== Accuracy per image ===")
        display(per_image)

        overall = pd.DataFrame({
            "metric": ["mean_accuracy_%", "fully_correct_images_%"],
            "value": [
                round(merged[correct_cols].mean().mean() * 100, 2),
                round(merged["all_correct"].mean() * 100, 2)
            ]
        })
        print("=== Overall Summary ===")
        display(overall)

        out_path = version_dir / f"accuracy_scores_{version_dir.name}.csv"
        merged.to_csv(out_path, index=False)
        print(f"✅ Saved to: {out_path}")