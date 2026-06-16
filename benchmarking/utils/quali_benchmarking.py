# vision_utils.py
from pathlib import Path
import torch
from qwen_vl_utils import process_vision_info  # adjust to your actual import


def describe_social_media_post_qwen(image_path, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": """Provide a full detailed description of this social media post image. Include:
- The profile name and any badges or indicators next to it
- The post text content
- Any images or charts shown in the post, incluiding the text in the image, the percentages and the colors.
- Engagement metrics such as likes, comments, shares
- Any other visual elements you can see"""
                }
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output[0]


def describe_social_media_post_gemma(image_path, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": """Provide a full detailed description of this social media post image. Include:
- The profile name and any badges or indicators next to it
- The post text content
- Any images or charts shown in the post, including the text in the image, the percentages and the colors.
- Engagement metrics such as likes, comments, shares
- Any other visual elements you can see"""
                }
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_path], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output[0]


def save_output_txt(
    image_path: str,
    description: str,
    folder_name: str,
    base_dir: Path,
    model_name: str,           # ← new
    prompt_name: str = "default"
):
    stem = Path(image_path).stem
    out_dir = base_dir / "outputs" / model_name / "qualitative" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{prompt_name}.txt"
    out_path.write_text(description, encoding="utf-8")
    print(f"✅ Saved to {out_path}")


def output_exists(
    image_path: str,
    folder_name: str,
    base_dir: Path,
    model_name: str,           # ← new
    prompt_name: str = "default"
) -> bool:
    stem = Path(image_path).stem
    out_path = base_dir / "outputs" / model_name / "qualitative" / folder_name / f"{stem}_{prompt_name}.txt"
    return out_path.exists()
    