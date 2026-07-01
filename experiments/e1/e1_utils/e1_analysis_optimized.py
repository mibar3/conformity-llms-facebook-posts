import json
import pandas as pd
from pathlib import Path
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REACTION_VALUES = [10, 100, 1000, 10000, 100000, 1000000]

SCALE_VALUES = [0, 10, 100, 1000, 10000, 100000, 1000000]

ADJACENT_PAIRS = [
    (c, i) for c in SCALE_VALUES for i in SCALE_VALUES
]


def load_json(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return json.loads(path.read_text())


def analyse_single(output_dir: Path, filename: str, like_answer: str = "like"):
    """Single image analysis — works for like/scroll and yes/no by passing like_answer."""
    results = load_json(output_dir / filename)
    df = pd.DataFrame(results)
    stem = Path(filename).stem

    print("\n" + "="*60)
    print(f"Single image analysis: {stem}")
    print("="*60)

    summary = pd.DataFrame({
        "metric": [
            f"overall_{like_answer}_rate_%",
            f"{like_answer}_rate_correct_%",
            f"{like_answer}_rate_incorrect_%"
        ],
        "value": [
            round((df["answer"] == like_answer).mean() * 100, 2),
            round((df[df["variant"] == "correct"]["answer"] == like_answer).mean() * 100, 2),
            round((df[df["variant"] == "incorrect"]["answer"] == like_answer).mean() * 100, 2),
        ]
    })
    print("=== Summary ===")
    display(summary)
    print("=== Per Image Results ===")
    display(df)

    out_path = output_dir / f"{stem.replace('results', 'analysis')}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")


def analyse_paired(output_dir: Path, filename: str):
    """Paired A/B analysis."""
    results = load_json(output_dir / filename)
    df = pd.DataFrame(results)
    stem = Path(filename).stem

    print("\n" + "="*60)
    print(f"Paired A/B analysis: {stem}")
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

    out_path = output_dir / f"{stem.replace('results', 'analysis')}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")


def analyse_metrics_single(output_dir: Path, filename: str, like_answer: str = "like"):
    """Metrics single image analysis — like rate per variant and per scale value."""
    results = load_json(output_dir / filename)
    df = pd.DataFrame(results)
    stem = Path(filename).stem

    df["scale_value"] = pd.Categorical(df["scale_value"], categories=REACTION_VALUES, ordered=True)
    df = df.sort_values(["scale_value", "variant", "image"])

    print("\n" + "="*60)
    print(f"Metrics single image analysis: {stem}")
    print("="*60)

    summary = pd.DataFrame({
        "metric": [
            f"overall_{like_answer}_rate_%",
            f"{like_answer}_rate_correct_%",
            f"{like_answer}_rate_incorrect_%"
        ],
        "value": [
            round((df["answer"] == like_answer).mean() * 100, 2),
            round((df[df["variant"] == "correct"]["answer"] == like_answer).mean() * 100, 2),
            round((df[df["variant"] == "incorrect"]["answer"] == like_answer).mean() * 100, 2),
        ]
    })
    print("=== Overall Summary ===")
    display(summary)

    per_scale = df.groupby("scale_value").apply(lambda g: pd.Series({
        "total_images": len(g),
        f"overall_{like_answer}_rate_%": round((g["answer"] == like_answer).mean() * 100, 2),
        f"{like_answer}_rate_correct_%": round((g[g["variant"] == "correct"]["answer"] == like_answer).mean() * 100, 2),
        f"{like_answer}_rate_incorrect_%": round((g[g["variant"] == "incorrect"]["answer"] == like_answer).mean() * 100, 2),
    })).reset_index()
    print("=== Rate per Scale Value ===")
    display(per_scale)

    print("=== Per Image Results ===")
    display(df)

    out_path = output_dir / f"{stem.replace('results', 'analysis')}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")


def analyse_metrics_paired(output_dir: Path, filename: str):
    results = load_json(output_dir / filename)
    df = pd.DataFrame(results)
    stem = Path(filename).stem

    # Create a readable label for each scale pair
    df["pair"] = df["correct_scale"].astype(str) + "_vs_" + df["incorrect_scale"].astype(str)
    pair_order = [f"{c}_vs_{i}" for c, i in ADJACENT_PAIRS]
    df["pair"] = pd.Categorical(df["pair"], categories=pair_order, ordered=True)
    df = df.sort_values(["pair", "num"])

    print("\n" + "="*60)
    print(f"Metrics paired A/B analysis: {stem}")
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

    per_pair = df.groupby("pair").apply(lambda g: pd.Series({
        "total_pairs": len(g),
        "liked_correct_%": round((g["liked_variant"] == "correct").mean() * 100, 2),
        "liked_incorrect_%": round((g["liked_variant"] == "incorrect").mean() * 100, 2),
        "invalid_%": round((g["liked_variant"] == "invalid").mean() * 100, 2),
    })).reset_index()
    print("=== Liked Correct Rate per Scale Pair ===")
    display(per_pair)

    print("=== Per Pair Results ===")
    display(df)

    out_path = output_dir / f"{stem.replace('results', 'analysis')}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")


def plot_ab_grid(output_dir: Path, filename: str, title: str):
    """Plot a grid of liked_correct_% for adjacent scale pairs."""
    results = load_json(output_dir / filename)
    df = pd.DataFrame(results)
    stem = Path(filename).stem

    scale_levels = [0, 10, 100, 1000, 10000, 100000, 1000000]
    scale_labels = ["0", "10", "100", "1K", "10K", "100K", "1M"]

    # Build grid
    n = len(scale_levels)
    grid = np.full((n, n), np.nan)

    for _, row in df.iterrows():
        c = scale_levels.index(row["correct_scale"])
        i = scale_levels.index(row["incorrect_scale"])
        # Aggregate liked_correct for this cell
        mask = (df["correct_scale"] == row["correct_scale"]) & (df["incorrect_scale"] == row["incorrect_scale"])
        liked_correct_pct = (df[mask]["liked_variant"] == "correct").mean() * 100
        grid[i][c] = round(liked_correct_pct, 1)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#d73027", "#f7f7f7", "#1a9850"])
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(grid[i][j]):
                color = "black"
                ax.text(j, i, f"{grid[i][j]:.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(scale_labels)
    ax.set_yticklabels(scale_labels)
    ax.invert_yaxis()  # to have it start at 0 and go up
    ax.set_xlabel("Correct post reactions", fontsize=12)
    ax.set_ylabel("Incorrect post reactions", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Liked correct post (%)")
    plt.tight_layout()

    out_path = output_dir / f"{stem}_grid.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"✅ Saved to: {out_path}")

def plot_ab_diff_grid(output_dir: Path, filename_a: str, filename_b: str, title: str):
    """Plot the difference in liked_correct_% between two paired A/B result files.
    Positive values (green) mean filename_a had higher correct preference than filename_b.
    Negative values (red) mean filename_b had higher correct preference than filename_a.
    """
    def build_grid(filename):
        results = load_json(output_dir / filename)
        df = pd.DataFrame(results)
        scale_levels = [0, 10, 100, 1000, 10000, 100000, 1000000]
        n = len(scale_levels)
        grid = np.full((n, n), np.nan)
        for _, row in df.iterrows():
            c = scale_levels.index(row["correct_scale"])
            i = scale_levels.index(row["incorrect_scale"])
            mask = (df["correct_scale"] == row["correct_scale"]) & (df["incorrect_scale"] == row["incorrect_scale"])
            grid[i][c] = round((df[mask]["liked_variant"] == "correct").mean() * 100, 1)
        return grid

    scale_labels = ["0", "10", "100", "1K", "10K", "100K", "1M"]
    grid_a = build_grid(filename_a)
    grid_b = build_grid(filename_b)
    diff_grid = grid_a - grid_b

    n = len(scale_labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#d73027", "#f7f7f7", "#1a9850"])
    im = ax.imshow(diff_grid, cmap=cmap, vmin=-100, vmax=100, aspect="auto")

    for i in range(n):
        for j in range(n):
            if not np.isnan(diff_grid[i][j]):
                val = diff_grid[i][j]
                sign = "+" if val > 0 else ""
                ax.text(j, i, f"{sign}{val:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(scale_labels)
    ax.set_yticklabels(scale_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Correct post reactions", fontsize=12)
    ax.set_ylabel("Incorrect post reactions", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label=f"Δ liked correct % ({Path(filename_a).stem} − {Path(filename_b).stem})")
    plt.tight_layout()

    stem = f"diff_{Path(filename_a).stem}_vs_{Path(filename_b).stem}"
    out_path = output_dir / f"{stem}_grid.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"✅ Saved to: {out_path}")