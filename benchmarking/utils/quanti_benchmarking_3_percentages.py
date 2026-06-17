import json
import torch
from pathlib import Path
from qwen_vl_utils import process_vision_info

PERCENTAGE_QUESTIONS = {
    "chart_percentages_pop": "What percentage is shown in the chart for Pop? Provide just the number.",
    "chart_percentages_latin": "What percentage is shown in the chart for Latin? Provide just the number.",
}

GROUND_TRUTH = {
    "chart_percentages_pop": "23.5",
    "chart_percentages_latin": "11.0",
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


def output_exists_json(image_name: str, base_dir: Path, experiment_name: str, model_name: str) -> bool:
    out_path = base_dir / "outputs" / model_name / "quantitative" / experiment_name / f"{image_name}.json"
    return out_path.exists()


def benchmark_image_percentages(image_path: str, image_name: str, model, processor, device, base_dir: Path, experiment_name: str, model_name: str, ask_fn):
    out_path = base_dir / "outputs" / model_name / "quantitative" / experiment_name / f"{image_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "image": image_name,
        "answers": {}
    }
    for question_key, question_text in PERCENTAGE_QUESTIONS.items():
        results["answers"][question_key] = ask_fn(image_path, question_text, model, processor, device)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {out_path}")