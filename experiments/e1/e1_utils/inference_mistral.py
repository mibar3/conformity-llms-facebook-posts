import torch

from e1_utils.logprob_scoring import extract_candidate_logprobs


def run_inference_mistral(messages: list, model, processor, device) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()


def run_inference_with_scores_mistral(messages: list, model, processor, device, candidates: list):
    """Same call as run_inference_mistral, but also returns each candidate's log-probability
    (read off the first-generated-token distribution — see logprob_scoring.py)."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
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
