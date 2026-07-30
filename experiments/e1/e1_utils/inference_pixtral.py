import torch
from PIL import Image

from e1_utils.logprob_scoring import extract_candidate_logprobs


def run_inference_pixtral(messages: list, model, processor, device) -> str:
    # Extract images in order from messages
    images = [
        Image.open(block["image"]).convert("RGB")
        for msg in messages
        for block in msg["content"]
        if block["type"] == "image"
    ]

    # Pixtral uses {"type": "image"} without a url/path in the chat template
    # Images are passed separately to the processor
    pixtral_messages = []
    for msg in messages:
        content = []
        for block in msg["content"]:
            if block["type"] == "image":
                content.append({"type": "image"})
            else:
                content.append(block)
        pixtral_messages.append({"role": msg["role"], "content": content})

    prompt = processor.apply_chat_template(
        pixtral_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=images if images else None,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False
        )

    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()


def run_inference_with_scores_pixtral(messages: list, model, processor, device, candidates: list):
    """Same call as run_inference_pixtral, but also returns each candidate's log-probability
    (read off the first-generated-token distribution — see logprob_scoring.py)."""
    images = [
        Image.open(block["image"]).convert("RGB")
        for msg in messages
        for block in msg["content"]
        if block["type"] == "image"
    ]

    pixtral_messages = []
    for msg in messages:
        content = []
        for block in msg["content"]:
            if block["type"] == "image":
                content.append({"type": "image"})
            else:
                content.append(block)
        pixtral_messages.append({"role": msg["role"], "content": content})

    prompt = processor.apply_chat_template(
        pixtral_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=images if images else None,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=16, do_sample=False,
            output_scores=True, return_dict_in_generate=True
        )

    output_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()

    candidate_logprobs = extract_candidate_logprobs(outputs.scores, processor.tokenizer, candidates)
    return answer, candidate_logprobs