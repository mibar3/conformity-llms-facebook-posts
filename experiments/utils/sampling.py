import json
import random
from pathlib import Path


def build_paired_sample(correct_dir: Path, incorrect_dir: Path, seed: int, sample_size: int, experiment_dir: Path) -> list:
    """
    Builds a 50:50 paired sample of correct/incorrect images.
    Saves selected_images.json to experiment_dir for reproducibility across LLMs.
    If it already exists, loads from it instead of resampling.
    """
    selection_path = experiment_dir / "selected_images.json"

    if selection_path.exists():
        print(f"📋 Loading existing selection from {selection_path}")
        selected = json.loads(selection_path.read_text())
        selected_numbers = selected["selected_numbers"]
    else:
        all_numbers = sorted([p.name.split("_")[0] for p in correct_dir.glob("*_remy_ashford_c.png")])
        print(f"Found {len(all_numbers)} images in {correct_dir}")
        random.seed(seed)
        selected_numbers = sorted(random.sample(all_numbers, sample_size))
        experiment_dir.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps({
            "seed": seed,
            "sample_size": sample_size,
            "selected_numbers": selected_numbers
        }, indent=2))
        print(f"✅ Saved selection to {selection_path}")

    missing = [num for num in selected_numbers if not (incorrect_dir / f"{num}_remy_ashford_i.png").exists()]
    if missing:
        raise FileNotFoundError(f"Missing in incorrect folder: {missing}")
    else:
        print("✅ All selected numbers verified in both correct and incorrect folders.")

    all_images = []
    for num in selected_numbers:
        all_images.append((f"{num}_correct", str(correct_dir / f"{num}_remy_ashford_c.png")))
        all_images.append((f"{num}_incorrect", str(incorrect_dir / f"{num}_remy_ashford_i.png")))

    print(f"Selected {len(selected_numbers)} pairs → {len(all_images)} images total")
    return all_images