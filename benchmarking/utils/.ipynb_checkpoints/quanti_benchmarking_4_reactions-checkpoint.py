import json
import torch
from pathlib import Path
from qwen_vl_utils import process_vision_info

PROMPT_VERSIONS = {
    "v1": "Does the text in the post accurately describe the chart? Reply with only 'correct' or 'incorrect'.",
    "v2": "Read the text in the post and look at the chart. Does the text correctly describe what the chart shows? Reply with only 'correct' or 'incorrect'.",
    "v3": "The post text claims one music genre is more popular than another. Look at the percentage values in the chart to verify this claim. If the genre described as more popular in the text has a higher percentage in the chart, reply 'correct'. If not, reply 'incorrect'.",
    "v4": "In the post, the text above the image makes a claim comparing the popularity of Pop and Latin. Identify the according values in the chart to verify this claim. If the claim matches the visualization reply 'correct'. If not, reply 'incorrect'.",
}


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
        "answers": {
            "post_claim_correct": ask_fn(image_path, prompt_text, model, processor, device)
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {out_path} [{prompt_version}]")