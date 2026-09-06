"""Uncertainty from a single greedy decode. sequence_probability is the
product of each step's chosen-token probability; 1 - that is the
uncertainty score used to gate the rescue step. first_token_entropy is a
secondary diagnostic (entropy of the first token's distribution)."""
import numpy as np


def sequence_probability(scores, gen_token_ids) -> float:
    if scores is None or gen_token_ids.shape[1] == 0:
        return 0.0
    import torch

    logp = 0.0
    for t, step_logits in enumerate(scores):
        step_log_probs = torch.log_softmax(step_logits[0], dim=-1)
        tok_id = int(gen_token_ids[0, t].item())
        logp += float(step_log_probs[tok_id].item())
    return float(np.exp(logp))


def first_token_entropy(scores) -> float:
    if not scores:
        return float("nan")
    import torch

    p = torch.softmax(scores[0][0], dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())
