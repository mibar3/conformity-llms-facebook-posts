import json
import torch
from pathlib import Path
from qwen_vl_utils import process_vision_info

QUESTIONS = {
    # Profile
    "profile_name": "What is the profile name shown in this post? Reply with just the name.",
    "profile_verification": "Is the profile verified? Reply with only 'yes' or 'no'.",
    # Engagement metrics
    "likes_count": "How many likes does this post have? Provide the full number, no letters.",
    "heart_reactions": "How many heart reactions does this post have? Provide the full number, no letters.",
    "haha_reactions": "How many haha reactions does this post have? Provide the full number, no letters.",
    "wow_reactions": "How many wow reactions does this post have? Provide the full number, no letters.",
    "sad_reactions": "How many sad reactions does this post have? Provide the full number, no letters.",
    "angry_reactions": "How many angry reactions does this post have? Provide the full number, no letters.",
    "comment_count": "How many comments does this post have? Provide the full number, no letters.",
    "shares_count": "How many shares does this post have? Provide the full number, no letters.",
    # Chart - values
    "chart_percentages_pop": "What percentage is shown in the chart for Pop? Provide just the number.",
    "chart_percentages_latin": "What percentage is shown in the chart for Latin? Provide just the number.",
    # Chart - color-genre mapping
    "chart_color_pop": "In the chart, what color represents Pop? Reply with just the color name.",
    "chart_color_latin": "In the chart, what color represents Latin? Reply with just the color name.",
    # Chart - dominant slice
    "chart_largest_slice": "Which music genre has the largest slice in the chart? Reply with just the genre name.",
}

PROMPT_VERSIONS = {
    "v1": "Does the text in the post accurately describe the chart? Reply with only 'correct' or 'incorrect'.",
    "v2": "Read the text in the post and look at the chart. Does the text correctly describe what the chart shows? Reply with only 'correct' or 'incorrect'.",
    "v3": "The post text claims one music genre is more popular than another. Look at the percentage values in the chart to verify this claim. If the genre described as more popular in the text has a higher percentage in the chart, reply 'correct'. If not, reply 'incorrect'.",
    "v4": "In the post, the text above the image makes a claim comparing the popularity of Pop and Latin. Identify the according values in the chart to verify this claim. If the claim matches the visualization reply 'correct'. If not, reply 'incorrect'.",
    "v5": "In the post, the text above the chart claims one genre is more popular than another. Find the percentage for Pop and the percentage for Latin in the chart. If the genre the text says is more popular has a numerically higher percentage, reply 'correct'. If not, reply 'incorrect'.",
    "v6": "Look at the text above the chart. It makes a claim about Pop and Latin popularity. Step 1: find the percentage value for Pop in the chart. Step 2: find the percentage value for Latin in the chart. Step 3: check if the claim in the text matches which one is higher. If it matches, reply 'correct'. If not, reply 'incorrect'."
}

TWO_CALL_PROMPT_VERSIONS = {
    "v7_two_call": "The chart shows Pop at {pop_pct}% and Latin at {latin_pct}%. The post text says '{claim}'. Based only on these numbers, is the claim correct? Reply with only 'correct' or 'incorrect'.",
    "v8_two_call": "Pop: {pop_pct}%. Latin: {latin_pct}%. The post claims: '{claim}'. Does the claim match which percentage is higher? Reply with only 'correct' or 'incorrect'.",
    "v9_two_call": "Given Pop = {pop_pct}% and Latin = {latin_pct}%, is the following statement true or false: '{claim}'. Reply with only 'correct' or 'incorrect'.",
}

CLAIM_EXTRACTION_PROMPT = "What claim does the post text make about Pop and Latin music popularity? Reply with just the claim sentence, nothing else."


def ask_question(image_path: str, question: str, model, processor, device) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
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
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def output_exists_json(image_name: str, prompt_version: str, base_dir: Path, experiment_name: str, model_name: str) -> bool:
    out_path = base_dir / "outputs" / model_name / "quantitative" / experiment_name / prompt_version / f"{image_name}.json"
    return out_path.exists()


def benchmark_image(image_path: str, image_name: str, prompt_version: str, prompt_text: str, model, processor, device, base_dir: Path, experiment_name: str, model_name: str, ask_fn):
    out_path = base_dir / "outputs" / model_name / "quantitative" / experiment_name / prompt_version / f"{image_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "image": image_name,
        "prompt_version": prompt_version,
        "prompt_text": prompt_text,
        "answers": {}
    }

    for question_key, question_text in {**QUESTIONS, "post_claim_correct": prompt_text}.items():
        results["answers"][question_key] = ask_fn(image_path, question_text, model, processor, device)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {out_path} [{prompt_version}]")


def extract_two_call_inputs(image_path: str, model, processor, device, ask_fn) -> dict:
    pop_pct = ask_fn(image_path, "What percentage is shown in the chart for Pop? Provide just the number.", model, processor, device)
    latin_pct = ask_fn(image_path, "What percentage is shown in the chart for Latin? Provide just the number.", model, processor, device)
    claim = ask_fn(image_path, CLAIM_EXTRACTION_PROMPT, model, processor, device)
    return {"pop_pct": pop_pct.strip(), "latin_pct": latin_pct.strip(), "claim": claim.strip()}


def benchmark_image_two_call(image_path: str, image_name: str, prompt_version: str, prompt_template: str, model, processor, device, base_dir: Path, experiment_name: str, model_name: str, ask_fn):
    out_path = base_dir / "outputs" / model_name / "quantitative" / experiment_name / prompt_version / f"{image_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extracted = extract_two_call_inputs(image_path, model, processor, device, ask_fn)
    filled_prompt = prompt_template.format(**extracted)

    results = {
        "image": image_name,
        "prompt_version": prompt_version,
        "prompt_text": filled_prompt,
        "extracted_inputs": extracted,
        "answers": {}
    }

    for question_key, question_text in {**QUESTIONS, "post_claim_correct": filled_prompt}.items():
        results["answers"][question_key] = ask_fn(image_path, question_text, model, processor, device)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {out_path} [{prompt_version}]")

def ask_question_gemma(image_path: str, question: str, model, processor, device) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_path], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]