"""Prompt templates for the gaze-target agent."""


def build_topk_prompt(vocab_str: str, k: int = 3) -> str:
    """First prompt: ask for the top-k ranked guesses in one generation call."""
    return f"""
The blue arrow helps you see where the person is looking. Its tip points to the object the person is focused on.
Select the object labels from this vocabulary list:

[{vocab_str}]

Output EXACTLY {k} labels separated by a comma, and nothing else. 
""".strip()


def build_rag_prompt(retrieved_labels: list, prior_top3: list, top_k: int) -> str:
    """Rescue prompt: retrieved labels merged with stage 1's own top-3 guesses as the candidate pool."""
    candidates = list(dict.fromkeys(retrieved_labels + prior_top3))  # dedup, keep order
    cand_str = ", ".join(candidates)

    return f"""Image {top_k + 1}: query image
The BLUE ARROW TIP shows
the target.
Choose EXACTLY THREE
labels from this list:
[{cand_str}]
Output EXACTLY three labels
separated by commas, and
nothing else."""
