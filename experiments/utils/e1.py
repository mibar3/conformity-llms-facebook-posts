import json
import torch
import random
from pathlib import Path
from qwen_vl_utils import process_vision_info


# --- Approch 1: Like or ignore - single image
LIKE_PROMPT_SINGLE = """You are shown a social media post.
You can either press the Like button or scroll past it.
Do you press Like on this post?
Reply with only 'like' or 'scroll'."""


def run_e1_single(all_images: list, model, processor, device, output_dir: Path):
    output_path = output_dir / "e1_results_single.json"

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
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": LIKE_PROMPT_SINGLE}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip().lower()

        results.append({
            "image": image_name,
            "variant": "correct" if "_correct" in image_name else "incorrect",
            "prompt": LIKE_PROMPT_SINGLE,
            "answer": answer
        })
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"✅ {image_name} → {answer}")

    print(f"\n✅ Done. Results saved to {output_path}")

# --- Approach 1: change in prompt - yes/no to liking
LIKE_PROMPT_YESNO = """You are shown a social media post.
You can either press the Like button or scroll past it.
Do you press Like on this post?
Reply with only 'yes' or 'no'."""

def run_e1_yesno(all_images: list, model, processor, device, output_dir: Path):
    output_path = output_dir / "e1_results_single_yesno.json"

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
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": LIKE_PROMPT_YESNO}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip().lower()

        results.append({
            "image": image_name,
            "variant": "correct" if "_correct" in image_name else "incorrect",
            "prompt": LIKE_PROMPT_YESNO,
            "answer": answer
        })
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"✅ {image_name} → {answer}")

    print(f"\n✅ Yes/No approach done. Results saved to {output_path}")
    
# --- Approach 2: like one of them -  paired A/B ---
LIKE_PROMPT_PAIR = """You are shown two social media posts, Post A and Post B.
You are only allowed to press the Like button on one of them.
Which post do you press Like on?
Reply with only 'A' or 'B'."""


def run_e1_paired(selected_numbers: list, correct_dir: Path, incorrect_dir: Path, model, processor, device, output_dir: Path, seed: int):
    output_path = output_dir / "e1_results_paired.json"

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
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Post A:"},
                    {"type": "image", "image": post_a_path},
                    {"type": "text", "text": "Post B:"},
                    {"type": "image", "image": post_b_path},
                    {"type": "text", "text": LIKE_PROMPT_PAIR}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip().upper()

        liked_variant = post_a_variant if answer == "A" else post_b_variant if answer == "B" else "invalid"

        results.append({
            "num": num,
            "post_a_variant": post_a_variant,
            "post_b_variant": post_b_variant,
            "prompt": LIKE_PROMPT_PAIR,
            "answer": answer,
            "liked_variant": liked_variant
        })
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"✅ Pair {num} → liked {liked_variant} (answered {answer})")

    print(f"\n✅ Paired approach done. Results saved to {output_path}")


# --- Metrics - single image ---------

REACTION_VALUES = [10, 100, 1000, 10000, 100000, 1000000]


def run_e1_metrics(selected_numbers: list, correct_base: Path, incorrect_base: Path, model, processor, device, output_dir: Path):
    output_path = output_dir / "e1_results_metrics.json"

    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {(r["image"]) for r in results}
        print(f"📋 Resuming — {len(already_done)} images already processed.")
    else:
        results = []
        already_done = set()

    output_dir.mkdir(parents=True, exist_ok=True)

    for scale_value in REACTION_VALUES:
        correct_dir = correct_base / str(scale_value)
        incorrect_dir = incorrect_base / str(scale_value)

        for num in selected_numbers:
            for variant, folder, suffix in [
                ("correct", correct_dir, "c"),
                ("incorrect", incorrect_dir, "i")
            ]:
                image_name = f"{num}_{variant}_{scale_value}"
                if image_name in already_done:
                    print(f"⏭ Skipping: {image_name}")
                    continue

                image_path = str(folder / f"{num}_remy_ashford_{suffix}.png")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": LIKE_PROMPT_SINGLE}
                        ]
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                answer = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip().lower()

                results.append({
                    "image": image_name,
                    "num": num,
                    "variant": variant,
                    "scale_value": scale_value,
                    "prompt": LIKE_PROMPT_SINGLE,
                    "answer": answer
                })
                output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                print(f"✅ {image_name} → {answer}")

    print(f"\n✅ Metrics approach done. Results saved to {output_path}")


# --- Metrics -- A/B paired
def run_e1_metrics_paired(selected_numbers: list, correct_base: Path, incorrect_base: Path, model, processor, device, output_dir: Path, seed: int):
    output_path = output_dir / "e1_results_metrics_paired.json"

    if output_path.exists():
        results = json.loads(output_path.read_text())
        already_done = {r["image"] for r in results}
        print(f"📋 Resuming — {len(already_done)} pairs already processed.")
    else:
        results = []
        already_done = set()

    output_dir.mkdir(parents=True, exist_ok=True)

    for scale_value in REACTION_VALUES:
        correct_dir = correct_base / str(scale_value)
        incorrect_dir = incorrect_base / str(scale_value)

        for num in selected_numbers:
            image_name = f"{num}_{scale_value}"
            if image_name in already_done:
                print(f"⏭ Skipping pair: {image_name}")
                continue

            correct_path = str(correct_dir / f"{num}_remy_ashford_c.png")
            incorrect_path = str(incorrect_dir / f"{num}_remy_ashford_i.png")

            # Randomise A/B per number AND scale value
            rng = random.Random(seed + int(num) + scale_value)
            if rng.random() < 0.5:
                post_a_path, post_a_variant = correct_path, "correct"
                post_b_path, post_b_variant = incorrect_path, "incorrect"
            else:
                post_a_path, post_a_variant = incorrect_path, "incorrect"
                post_b_path, post_b_variant = correct_path, "correct"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Post A:"},
                        {"type": "image", "image": post_a_path},
                        {"type": "text", "text": "Post B:"},
                        {"type": "image", "image": post_b_path},
                        {"type": "text", "text": LIKE_PROMPT_PAIR}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip().upper()

            liked_variant = post_a_variant if answer == "A" else post_b_variant if answer == "B" else "invalid"

            results.append({
                "image": image_name,
                "num": num,
                "scale_value": scale_value,
                "post_a_variant": post_a_variant,
                "post_b_variant": post_b_variant,
                "prompt": LIKE_PROMPT_PAIR,
                "answer": answer,
                "liked_variant": liked_variant
            })
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"✅ Pair {image_name} → liked {liked_variant} (answered {answer})")

    print(f"\n✅ Metrics paired approach done. Results saved to {output_path}")