import torch
from PIL import Image


def run_inference_ovis(messages: list, model, processor, device) -> str:
    text_tokenizer = model.text_tokenizer

    pil_messages = []
    for msg in messages:
        content = []
        for block in msg["content"]:
            if block["type"] == "image":
                content.append({"type": "image", "image": Image.open(block["image"]).convert("RGB")})
            else:
                content.append(block)
        pil_messages.append({"role": msg["role"], "content": content})

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=pil_messages,
        add_generation_prompt=True,
        enable_thinking=False
    )

    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=model.dtype, device=model.device)

    # Merge multimodal embeddings manually
    inputs_embeds = model.merge_multimodal(
        input_ids=input_ids,
        pixel_values=pixel_values,
        grid_thws=grid_thws
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to(device=model.device)

    with torch.no_grad():
        output_ids = model.llm.generate(
            inputs=None,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.eos_token_id,
        )

    output_ids = output_ids[0][input_ids.shape[1]:]
    return text_tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()