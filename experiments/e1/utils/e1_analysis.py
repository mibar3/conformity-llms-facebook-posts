import json
import pandas as pd
from pathlib import Path
from IPython.display import display


def load_json(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return json.loads(path.read_text())


def analyse_baseline_single(output_dir: Path):
    """Approach 1: single image like/scroll."""
    results = load_json(output_dir / "e1_results_baseline.json")
    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("Approach: Single image (like/scroll)")
    print("="*60)

    summary = pd.DataFrame({
        "metric": [
            "overall_like_rate_%",
            "like_rate_correct_%",
            "like_rate_incorrect_%"
        ],
        "value": [
            round((df["answer"] == "like").mean() * 100, 2),
            round((df[df["variant"] == "correct"]["answer"] == "like").mean() * 100, 2),
            round((df[df["variant"] == "incorrect"]["answer"] == "like").mean() * 100, 2),
        ]
    })
    print("=== Summary ===")
    display(summary)
    print("=== Per Image Results ===")
    display(df)

    out_path = output_dir / "e1_analysis_single.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")


def analyse_yesno(output_dir: Path):
    """Approach 1: single image yes/no."""
    results = load_json(output_dir / "e1_results_single_yesno.json")
    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("Approach: Single image (yes/no)")
    print("="*60)

    summary = pd.DataFrame({
        "metric": [
            "overall_yes_rate_%",
            "yes_rate_correct_%",
            "yes_rate_incorrect_%"
        ],
        "value": [
            round((df["answer"] == "yes").mean() * 100, 2),
            round((df[df["variant"] == "correct"]["answer"] == "yes").mean() * 100, 2),
            round((df[df["variant"] == "incorrect"]["answer"] == "yes").mean() * 100, 2),
        ]
    })
    print("=== Summary ===")
    display(summary)
    print("=== Per Image Results ===")
    display(df)

    out_path = output_dir / "e1_analysis_yesno.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")


def analyse_paired(output_dir: Path):
    """Approach 2: paired A/B."""
    results = load_json(output_dir / "e1_results_paired.json")
    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("Approach 2: Paired A/B")
    print("="*60)

    summary = pd.DataFrame({
        "metric": [
            "liked_correct_%",
            "liked_incorrect_%",
            "invalid_answer_%"
        ],
        "value": [
            round((df["liked_variant"] == "correct").mean() * 100, 2),
            round((df["liked_variant"] == "incorrect").mean() * 100, 2),
            round((df["liked_variant"] == "invalid").mean() * 100, 2),
        ]
    })
    print("=== Summary ===")
    display(summary)
    print("=== Per Pair Results ===")
    display(df)

    out_path = output_dir / "e1_analysis_paired.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")


def analyse_metrics(output_dir: Path):
    """Approach 1 on metrics folders — like rate per variant and per scale value."""
    results = load_json(output_dir / "e1_results_metrics.json")
    df = pd.DataFrame(results)

    REACTION_VALUES = [10, 100, 1000, 10000, 100000, 1000000]
    df["scale_value"] = pd.Categorical(df["scale_value"], categories=REACTION_VALUES, ordered=True)
    df = df.sort_values(["scale_value", "variant", "image"])

    print("\n" + "="*60)
    print("Approach 1 on metrics: like/scroll per scale value")
    print("="*60)

    # --- Overall summary ---
    summary = pd.DataFrame({
        "metric": [
            "overall_like_rate_%",
            "like_rate_correct_%",
            "like_rate_incorrect_%"
        ],
        "value": [
            round((df["answer"] == "like").mean() * 100, 2),
            round((df[df["variant"] == "correct"]["answer"] == "like").mean() * 100, 2),
            round((df[df["variant"] == "incorrect"]["answer"] == "like").mean() * 100, 2),
        ]
    })
    print("=== Overall Summary ===")
    display(summary)

    # --- Breakdown by scale value ---
    per_scale = df.groupby("scale_value").apply(lambda g: pd.Series({
        "total_images": len(g),
        "overall_like_rate_%": round((g["answer"] == "like").mean() * 100, 2),
        "like_rate_correct_%": round((g[g["variant"] == "correct"]["answer"] == "like").mean() * 100, 2),
        "like_rate_incorrect_%": round((g[g["variant"] == "incorrect"]["answer"] == "like").mean() * 100, 2),
    })).reset_index()
    print("=== Like Rate per Scale Value ===")
    display(per_scale)

    print("=== Per Image Results ===")
    display(df)

    out_path = output_dir / "e1_analysis_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")

def analyse_metrics_paired(output_dir: Path):
    """Approach 2 on metrics folders — paired A/B per scale value."""
    results = load_json(output_dir / "e1_results_metrics_paired.json")
    df = pd.DataFrame(results)

    REACTION_VALUES = [10, 100, 1000, 10000, 100000, 1000000]
    df["scale_value"] = pd.Categorical(df["scale_value"], categories=REACTION_VALUES, ordered=True)
    df = df.sort_values(["scale_value", "num"])

    print("\n" + "="*60)
    print("Approach 2 on metrics: paired A/B per scale value")
    print("="*60)

    summary = pd.DataFrame({
        "metric": [
            "overall_liked_correct_%",
            "overall_liked_incorrect_%",
            "invalid_answer_%"
        ],
        "value": [
            round((df["liked_variant"] == "correct").mean() * 100, 2),
            round((df["liked_variant"] == "incorrect").mean() * 100, 2),
            round((df["liked_variant"] == "invalid").mean() * 100, 2),
        ]
    })
    print("=== Overall Summary ===")
    display(summary)

    per_scale = df.groupby("scale_value").apply(lambda g: pd.Series({
        "total_pairs": len(g),
        "liked_correct_%": round((g["liked_variant"] == "correct").mean() * 100, 2),
        "liked_incorrect_%": round((g["liked_variant"] == "incorrect").mean() * 100, 2),
        "invalid_%": round((g["liked_variant"] == "invalid").mean() * 100, 2),
    })).reset_index()
    print("=== Liked Correct Rate per Scale Value ===")
    display(per_scale)

    print("=== Per Pair Results ===")
    display(df)

    out_path = output_dir / "e1_analysis_metrics_paired.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")