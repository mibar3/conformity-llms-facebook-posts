import json
import random
from pathlib import Path

LIKE_PROMPT_SINGLE = """You are shown a social media post.
You can either press the Like button or scroll past it.
Do you press Like on this post?
Reply with only 'like' or 'scroll'."""

LIKE_PROMPT_YESNO = """You are shown a social media post.
You can either press the Like button or scroll past it.
Do you press Like on this post?
Reply with only 'yes' or 'no'."""

LIKE_PROMPT_PAIR = """You are shown two social media posts, Post A and Post B.
You are only allowed to press the Like button on one of them.
Which post do you press Like on?
Reply with only 'A' or 'B'."""

REACTION_VALUES = [10, 100, 1000, 10000, 100000, 1000000]
SCALE_VALUES = [0, 10, 100, 1000, 10000, 100000, 1000000]
ADJACENT_PAIRS = [(c, i) for c in SCALE_VALUES for i in SCALE_VALUES]


def run_e1_baseline(all_images: list, model, processor, device, output_dir: Path, prompt: str, output_filename: str, inference_fn):
    output_path = output_dir / output_filename
    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {r["image"] for r in results}
        print(f"📋 Resuming — {len(already_done)} images already processed.")
    else:
        results = []
        already_done = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_name, image_path in all_images:
        if image_name in already_done:
            print(f"⏭ Skipping: {image_name}")
            continue
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]
        answer = inference_fn(messages, model, processor, device)
        results.append({
            "image": image_name,
            "variant": "correct" if "_correct" in image_name else "incorrect",
            "prompt": prompt,
            "answer": answer
        })
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"✅ {image_name} → {answer}")
    print(f"\n✅ Done. Results saved to {output_path}")


def run_e1_baseline_paired(selected_numbers: list, correct_dir: Path, incorrect_dir: Path, model, processor, device, output_dir: Path, seed: int, prompt: str, output_filename: str, inference_fn):
    output_path = output_dir / output_filename
    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {r["num"] for r in results}
        print(f"📋 Resuming — {len(already_done)} pairs already processed.")
    else:
        results = []
        already_done = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for num in selected_numbers:
        if num in already_done:
            print(f"⏭ Skipping pair: {num}")
            continue
        correct_path = str(correct_dir / f"{num}_remy_ashford_c.png")
        incorrect_path = str(incorrect_dir / f"{num}_remy_ashford_i.png")
        rng = random.Random(seed + int(num))
        if rng.random() < 0.5:
            post_a_path, post_a_variant = correct_path, "correct"
            post_b_path, post_b_variant = incorrect_path, "incorrect"
        else:
            post_a_path, post_a_variant = incorrect_path, "incorrect"
            post_b_path, post_b_variant = correct_path, "correct"
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Post A:"},
                {"type": "image", "image": post_a_path},
                {"type": "text", "text": "Post B:"},
                {"type": "image", "image": post_b_path},
                {"type": "text", "text": prompt}
            ]}
        ]
        answer = inference_fn(messages, model, processor, device).upper()
        liked_variant = post_a_variant if answer == "A" else post_b_variant if answer == "B" else "invalid"
        results.append({
            "num": num,
            "post_a_variant": post_a_variant,
            "post_b_variant": post_b_variant,
            "prompt": prompt,
            "answer": answer,
            "liked_variant": liked_variant
        })
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"✅ Pair {num} → liked {liked_variant} (answered {answer})")
    print(f"\n✅ Paired approach done. Results saved to {output_path}")


def run_e1_metrics(selected_numbers: list, correct_base: Path, incorrect_base: Path, model, processor, device, output_dir: Path, prompt: str, output_filename: str, inference_fn):
    output_path = output_dir / output_filename
    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {r["image"] for r in results}
        print(f"📋 Resuming — {len(already_done)} images already processed.")
    else:
        results = []
        already_done = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for scale_value in REACTION_VALUES:
        correct_dir = correct_base / str(scale_value)
        incorrect_dir = incorrect_base / str(scale_value)
        for num in selected_numbers:
            for variant, folder, suffix in [("correct", correct_dir, "c"), ("incorrect", incorrect_dir, "i")]:
                image_name = f"{num}_{variant}_{scale_value}"
                if image_name in already_done:
                    print(f"⏭ Skipping: {image_name}")
                    continue
                image_path = str(folder / f"{num}_remy_ashford_{suffix}.png")
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt}
                    ]}
                ]
                answer = inference_fn(messages, model, processor, device)
                results.append({
                    "image": image_name, "num": num, "variant": variant,
                    "scale_value": scale_value, "prompt": prompt, "answer": answer
                })
                output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                print(f"✅ {image_name} → {answer}")
    print(f"\n✅ Metrics done. Results saved to {output_path}")


def run_e1_metrics_paired(selected_numbers: list, correct_base: Path, incorrect_base: Path, model, processor, device, output_dir: Path, seed: int, prompt: str, output_filename: str, inference_fn, baseline_correct_dir: Path = None, baseline_incorrect_dir: Path = None):
    output_path = output_dir / output_filename
    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {r["image"] for r in results}
        print(f"📋 Resuming — {len(already_done)} pairs already processed.")
    else:
        results = []
        already_done = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    if baseline_correct_dir is None:
        baseline_correct_dir = correct_base.parents[1]
    if baseline_incorrect_dir is None:
        baseline_incorrect_dir = incorrect_base.parents[1]

    for correct_scale, incorrect_scale in ADJACENT_PAIRS:
        correct_dir = baseline_correct_dir if correct_scale == 0 else correct_base / str(correct_scale)
        incorrect_dir = baseline_incorrect_dir if incorrect_scale == 0 else incorrect_base / str(incorrect_scale)

        for num in selected_numbers:
            image_name = f"{num}_correct{correct_scale}_vs_incorrect{incorrect_scale}"
            if image_name in already_done:
                print(f"⏭ Skipping pair: {image_name}")
                continue
            correct_path = str(correct_dir / f"{num}_remy_ashford_c.png")
            incorrect_path = str(incorrect_dir / f"{num}_remy_ashford_i.png")
            rng = random.Random(seed + int(num) + correct_scale + incorrect_scale)
            if rng.random() < 0.5:
                post_a_path, post_a_variant = correct_path, "correct"
                post_b_path, post_b_variant = incorrect_path, "incorrect"
            else:
                post_a_path, post_a_variant = incorrect_path, "incorrect"
                post_b_path, post_b_variant = correct_path, "correct"
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "Post A:"},
                    {"type": "image", "image": post_a_path},
                    {"type": "text", "text": "Post B:"},
                    {"type": "image", "image": post_b_path},
                    {"type": "text", "text": prompt}
                ]}
            ]
            answer = inference_fn(messages, model, processor, device).upper()
            liked_variant = post_a_variant if answer == "A" else post_b_variant if answer == "B" else "invalid"
            results.append({
                "image": image_name, "num": num,
                "correct_scale": correct_scale, "incorrect_scale": incorrect_scale,
                "post_a_variant": post_a_variant, "post_b_variant": post_b_variant,
                "prompt": prompt, "answer": answer, "liked_variant": liked_variant
            })
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"✅ {image_name} → liked {liked_variant} (answered {answer})")
    print(f"\n✅ Metrics paired done. Results saved to {output_path}")