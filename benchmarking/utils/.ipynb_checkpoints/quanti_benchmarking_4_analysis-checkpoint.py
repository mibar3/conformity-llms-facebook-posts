import json
import pandas as pd
from pathlib import Path
from IPython.display import display

REACTION_VALUES = [10, 100, 1000, 10000, 100000, 1000000]


def run_accuracy_analysis(base_dir: Path, experiment_name: str):
    test_dir = base_dir / "outputs/quantitative" / experiment_name
    version_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    for version_dir in version_dirs:
        records = []
        for f in version_dir.glob("*.json"):
            data = json.loads(f.read_text())
            image_name = data["image"]
            parts = image_name.split("_")
            variant = "correct" if "_correct_" in image_name else "incorrect"
            scale_value = int(parts[-1])
            prediction = data["answers"]["post_claim_correct"].strip().lower()
            records.append({
                "image": image_name,
                "scale_value": scale_value,
                "variant": variant,
                "ground_truth": variant,
                "prediction": prediction,
                "correct": prediction == variant
            })

        if not records:
            print(f"⚠️ No JSON files found in {version_dir.name}, skipping.")
            continue

        df = pd.DataFrame(records)
        df["scale_value"] = pd.Categorical(df["scale_value"], categories=REACTION_VALUES, ordered=True)
        df = df.sort_values(["scale_value", "variant", "image"])

        prompt_text = json.loads(list(version_dir.glob("*.json"))[0].read_text()).get("prompt_text", "unknown")
        print(f"\n{'='*60}")
        print(f"Prompt Version : {version_dir.name}")
        print(f"Prompt Text    : {prompt_text}")
        print(f"{'='*60}")

        overall = pd.DataFrame({
            "metric": [
                "overall_accuracy_%",
                "accuracy_correct_posts_%",
                "accuracy_incorrect_posts_%",
            ],
            "value": [
                round(df["correct"].mean() * 100, 2),
                round(df[df["variant"] == "correct"]["correct"].mean() * 100, 2),
                round(df[df["variant"] == "incorrect"]["correct"].mean() * 100, 2),
            ]
        })
        print("=== Overall Summary ===")
        display(overall)

        per_scale = df.groupby("scale_value").apply(lambda g: pd.Series({
            "total_images": len(g),
            "overall_accuracy_%": round(g["correct"].mean() * 100, 2),
            "correct_posts_accuracy_%": round(g[g["variant"] == "correct"]["correct"].mean() * 100, 2),
            "incorrect_posts_accuracy_%": round(g[g["variant"] == "incorrect"]["correct"].mean() * 100, 2),
        })).reset_index()
        print("=== Accuracy per Reaction Level ===")
        display(per_scale)

        print("=== Per Image Results ===")
        display(df)

        out_path = version_dir / f"accuracy_scores_{version_dir.name}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved to: {out_path}")