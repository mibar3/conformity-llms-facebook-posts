import torch

"""
handles both cases: single image (one path in the list) for baseline/metrics runs, and two images (correct + incorrect) for the paired A/B runs 
"""

def run_inference_gemma(messages: list, model, processor, device) -> str:
    # Extract all image paths from the messages, in order
    image_paths = [
        block["image"]
        for msg in messages
        for block in msg["content"]
        if block["type"] == "image"
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=image_paths, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()