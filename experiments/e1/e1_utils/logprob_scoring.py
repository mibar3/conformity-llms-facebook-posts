import math

import torch.nn.functional as F

"""
Forced-choice log-probability reading for the isolated-judgment protocol (like/scroll,
yes/no). Complements run_inference_* (which only returns the greedy-decoded string) for
the models/conditions where decoding degenerates to a constant 0%/100% answer — reading
the probabilities the model actually assigned exposes how close the call really was,
instead of just the final decision.

No extra model calls and no prompt changes: generate() already computes a probability for
every possible next word before picking the winner (that's how greedy decoding works). We
just ask it to hand that distribution back (output_scores=True, return_dict_in_generate=True
on the same generate() call run_inference_* already makes) and read off the two candidate
words from it, instead of only keeping the word that happened to win.

Only reads the FIRST generated token, so this only gives an exact answer when each
candidate is a single token in the model's tokenizer (near-certain for "like"/"scroll"/
"yes"/"no", but not verified against a live model yet — the assert below will fail loudly
on the first server run if that assumption is ever wrong for one of the 4 model families).
"""


def extract_candidate_logprobs(scores, tokenizer, candidates: list) -> dict:
    """
    scores: the `.scores` field from generate(..., output_scores=True,
    return_dict_in_generate=True) — a tuple with one [batch, vocab_size] tensor per
    generated token. scores[0] is the distribution over the full vocabulary for the first
    generated token, i.e. the exact moment the model "decides" like vs scroll (or yes vs
    no). Batch size 1 assumed throughout this project.
    """
    first_token_logits = scores[0][0]
    logprobs_all = F.log_softmax(first_token_logits.float(), dim=-1)

    candidate_logprobs = {}
    for cand in candidates:
        token_ids = tokenizer(" " + cand.strip(), add_special_tokens=False).input_ids
        assert len(token_ids) == 1, (
            f"{cand!r} tokenizes to {len(token_ids)} tokens for this model, not 1 — "
            "this scorer only reads the first generated token, so it can't score a "
            "multi-token candidate. Flag this and extend extract_candidate_logprobs."
        )
        candidate_logprobs[cand.strip().lower()] = logprobs_all[token_ids[0]].item()

    max_lp = max(candidate_logprobs.values())
    denom = sum(math.exp(lp - max_lp) for lp in candidate_logprobs.values())
    return {
        c: {"logprob": lp, "prob_forced_choice": math.exp(lp - max_lp) / denom}
        for c, lp in candidate_logprobs.items()
    }
