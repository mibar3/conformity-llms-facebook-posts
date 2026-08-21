"""
Cross-model figures for the main E1 study.

Deliberately does NOT duplicate what already exists:

  * per-model 7x7 grids            -> e1_utils.e1_analysis_optimized.plot_ab_grid
  * per-model correct-vs-correct   -> e1_utils.e1_analysis_optimized.plot_cc_grid
  * condition-vs-condition diffs   -> e1_utils.e1_analysis_optimized.plot_ab_diff_grid
  * 2x3 grid overviews per signal  -> statistical_analysis/grid_overview_figures.ipynb

What it adds is the cross-model summaries and, above all, the **position diagnostic** — the check
that a paired-choice result cannot be characterised by conditioning on one variable alone. Three
findings in this project were written up wrong for exactly that reason (see THESIS.md's 2026-08-19
and 2026-08-21 changelog entries), and until now the check existed only as numbers in a script,
never as a figure.

Lives in statistical_analysis/ rather than e1_utils/ so that the module every experiment notebook
imports stays untouched; nothing here is needed at inference time.

All arithmetic is plain Python so it can be verified without numpy/matplotlib; only the drawing
imports them.
"""
import json
from pathlib import Path

SCALE_LABELS = ["0", "10", "100", "1K", "10K", "100K", "1M"]

# same 2x3 ordering as grid_overview_figures.ipynb
ROSTER = [("Gemma-12B", "gemma4-12b"), ("Qwen3-VL-8B", "qwen3-vl-8b"),
          ("Ministral-3-14B", "ministral-3-14b"), ("Gemma-E4B", "gemma4-e4b"),
          ("Qwen3-VL-4B", "qwen3-vl-4b"), ("Ministral-3-8B", "ministral-3-8b")]

CONDITIONS = [("metrics", "e1_results_metrics_paired.json", "Metrics (all reaction types)"),
              ("likes_only_noise", "e1_results_likes_only_noise_paired.json", "Likes only (with noise)")]


def _load(repo_root: Path, slug: str, filename: str):
    p = Path(repo_root) / "experiments/e1" / slug / "outputs" / filename
    return [r for r in json.loads(p.read_text()) if r["answer"] in ("A", "B")]


def _pct(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows) * 100 if rows else None


def competence_collapse_series(repo_root: Path, filename: str, models=None):
    """Per model: tied-engagement accuracy ("competence"), accuracy when the correct post is
    disadvantaged ("collapse"), and the tied-trial A-rate that says whether the first number
    means anything at all."""
    out = []
    for label, slug in (models or ROSTER):
        rows = _load(repo_root, slug, filename)
        tied = [r for r in rows if r["correct_scale"] == r["incorrect_scale"]]
        above = [r for r in rows if r["incorrect_scale"] > r["correct_scale"]]
        below = [r for r in rows if r["correct_scale"] > r["incorrect_scale"]]
        correct = lambda r: r["liked_variant"] == "correct"
        out.append({
            "label": label, "slug": slug,
            "diagonal": _pct(tied, correct),
            "pressure": _pct(above, correct),
            "support": _pct(below, correct),
            "tied_a_rate": _pct(tied, lambda r: r["answer"] == "A"),
            "n_tied": len(tied),
        })
    return out


def position_diagnostic_series(repo_root: Path, filename: str, models=None):
    """The cross-tab. For off-diagonal trials, how much does the answer swing with WHERE the
    correct post sits, versus where the MORE-ENGAGED post sits?

    A model tracking correctness shows a large `swing_correct`; one tracking engagement shows a
    large `swing_engagement`; one answering by position shows neither, plus an extreme A-rate.
    Reading only one of these is what produced the mischaracterisations recorded in THESIS.md.
    """
    out = []
    for label, slug in (models or ROSTER):
        rows = [r for r in _load(repo_root, slug, filename)
                if r["correct_scale"] != r["incorrect_scale"]]
        a = lambda rs: _pct(rs, lambda r: r["answer"] == "A")

        corr_a = [r for r in rows if r["post_a_variant"] == "correct"]
        corr_b = [r for r in rows if r["post_a_variant"] != "correct"]

        def more_engaged_in_a(r):
            hi = "correct" if r["correct_scale"] > r["incorrect_scale"] else "incorrect"
            return r["post_a_variant"] == hi
        eng_a = [r for r in rows if more_engaged_in_a(r)]
        eng_b = [r for r in rows if not more_engaged_in_a(r)]

        out.append({
            "label": label, "slug": slug, "n": len(rows),
            "a_rate": a(rows),
            "swing_correct": abs(a(corr_a) - a(corr_b)),
            "swing_engagement": abs(a(eng_a) - a(eng_b)),
        })
    return out


def plot_competence_vs_collapse(repo_root: Path, filename: str, cond_title: str,
                                models=None, save_path: Path = None):
    """Dumbbell: each model's tied-engagement accuracy against its accuracy under pressure.
    Models whose tied figure is a positional artifact are drawn hollow and flagged."""
    import matplotlib.pyplot as plt

    data = competence_collapse_series(repo_root, filename, models)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, d in enumerate(data):
        artifact = max(d["tied_a_rate"], 100 - d["tied_a_rate"]) >= 90
        ax.plot([d["pressure"], d["diagonal"]], [i, i], color="#bbbbbb", lw=2, zorder=1)
        ax.scatter(d["diagonal"], i, s=130, zorder=2, color="white" if artifact else "#2a78d6",
                   edgecolor="#2a78d6", linewidth=2,
                   label="tied engagement" if i == 0 else None)
        ax.scatter(d["pressure"], i, s=130, zorder=2, color="#e34948",
                   label="correct post disadvantaged" if i == 0 else None)
        if artifact:
            ax.annotate(f"tied score is positional ({d['tied_a_rate']:.0f}% one slot)",
                        (d["diagonal"], i), textcoords="offset points", xytext=(12, 0),
                        fontsize=8, color="#777777", va="center")

    ax.axvline(50, ls="--", color="#888888", lw=1)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d["label"] for d in data])
    ax.invert_yaxis()
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Chose the correct post (%)", fontsize=12)
    ax.set_title(f"Competence vs. collapse — {cond_title}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved to: {save_path}")
    plt.show()

    for d in data:
        if max(d["tied_a_rate"], 100 - d["tied_a_rate"]) >= 90:
            print(f"⚠ {d['label']}: tied-engagement {d['diagonal']:.1f}% is NOT competence — "
                  f"it answers one slot on {max(d['tied_a_rate'], 100-d['tied_a_rate']):.1f}% "
                  f"of {d['n_tied']} tied trials.")


def plot_position_diagnostic(repo_root: Path, filename: str, cond_title: str,
                             models=None, save_path: Path = None):
    """What is each model actually responding to: correctness, engagement, or the slot?"""
    import numpy as np
    import matplotlib.pyplot as plt

    data = position_diagnostic_series(repo_root, filename, models)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.22, [d["swing_correct"] for d in data], 0.42, color="#2a78d6",
           label="swing with where the CORRECT post sits")
    ax.bar(x + 0.22, [d["swing_engagement"] for d in data], 0.42, color="#e34948",
           label="swing with where the MORE-ENGAGED post sits")
    for i, d in enumerate(data):
        ax.annotate(f"A-rate\n{d['a_rate']:.0f}%", (i, 2), ha="center", va="bottom",
                    fontsize=8, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([d["label"] for d in data], fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Swing in % answering 'A' (percentage points)", fontsize=11)
    ax.set_title(f"What is the model tracking? — {cond_title}\n"
                 "off-diagonal trials only; tall bar = that cue drives the answer",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved to: {save_path}")
    plt.show()


def plot_cc_overview(repo_root: Path, filename: str, cond_title: str,
                     models=None, save_path: Path = None):
    """2x3 of the correct-vs-correct control: both posts correct, so only engagement can decide.
    The diagonal is left blank because both posts are identical there."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    scales = [0, 10, 100, 1000, 10000, 100000, 1000000]
    n = len(scales)
    cmap = mcolors.LinearSegmentedColormap.from_list("red_gray_blue",
                                                     ["#e34948", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    flat = axes.flatten()
    im = None
    for ax, (label, slug) in zip(flat, (models or ROSTER)):
        rows = _load(repo_root, slug, filename)
        grid = np.full((n, n), np.nan)
        for i, sa in enumerate(scales):
            for j, sb in enumerate(scales):
                if sa == sb:
                    continue
                cell = [r for r in rows if r["scale_a"] == sa and r["scale_b"] == sb]
                if cell:
                    grid[j][i] = _pct(cell, lambda r: r["liked_higher_engagement"] is True)
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(SCALE_LABELS, fontsize=8, rotation=45)
        ax.set_yticklabels(SCALE_LABELS, fontsize=8)
        ax.invert_yaxis()
    fig.supxlabel("Post A reactions (both posts correct)", fontsize=13)
    fig.supylabel("Post B reactions (both posts correct)", fontsize=13)
    fig.suptitle(f"% chose the more-engaged post — {cond_title}",
                 fontsize=16, fontweight="bold", y=1.02)
    cbar = fig.colorbar(im, ax=flat.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Chose the more-engaged post (%)   —   gray = 50%, chance", fontsize=11)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved to: {save_path}")
    plt.show()


SINGLE = [("baseline", "e1_results_baseline.json"), ("likes_only", "e1_results_likes_only.json"),
          ("likes_only_noise", "e1_results_likes_only_noise.json"),
          ("metrics", "e1_results_metrics.json")]


def plot_single_image_saturation(repo_root: Path, like_answer: str = "like",
                                 files=None, models=None, save_path: Path = None):
    """The isolated-judgment protocol saturates to 0% or 100% for most models (Section 5.3).
    One dot per model x condition: anything pinned at an edge carries no information."""
    import matplotlib.pyplot as plt

    files = files or SINGLE
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2a78d6", "#e34948", "#3fa34d", "#c46a1f"]
    for k, (cond, fname) in enumerate(files):
        xs, ys = [], []
        for i, (label, slug) in enumerate(models or ROSTER):
            p = Path(repo_root) / "experiments/e1" / slug / "outputs" / fname
            if not p.exists():
                continue
            rows = [r for r in json.loads(p.read_text())
                    if str(r["answer"]).strip().lower() in (like_answer, "scroll", "yes", "no")]
            if not rows:
                continue
            xs.append(_pct(rows, lambda r: str(r["answer"]).strip().lower() == like_answer))
            ys.append(i)
        ax.scatter(xs, ys, s=90, color=colors[k % len(colors)], label=cond, alpha=0.85, zorder=2)
    ax.axvline(50, ls="--", color="#888888", lw=1)
    ax.set_yticks(range(len(models or ROSTER)))
    ax.set_yticklabels([l for l, _ in (models or ROSTER)])
    ax.invert_yaxis()
    ax.set_xlim(-3, 103)
    ax.set_xlabel(f"Answered '{like_answer}' (%)", fontsize=12)
    ax.set_title("Isolated-judgment protocol: saturation by model and condition\n"
                 "dots pinned at 0% or 100% carry no information",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved to: {save_path}")
    plt.show()
