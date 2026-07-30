import torch
from qwen_vl_utils import process_vision_info

from e1_utils.logprob_scoring import extract_candidate_logprobs

def run_inference_qwen(messages: list, model, processor, device) -> str:
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
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()


def run_inference_with_scores_qwen(messages: list, model, processor, device, candidates: list):
    """Same call as run_inference_qwen, but also returns each candidate's log-probability
    (read off the first-generated-token distribution — see logprob_scoring.py)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=16, do_sample=False,
            output_scores=True, return_dict_in_generate=True
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()

    candidate_logprobs = extract_candidate_logprobs(outputs.scores, processor.tokenizer, candidates)
    return answer, candidate_logprobs