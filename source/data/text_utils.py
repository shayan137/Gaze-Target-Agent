"""Text normalization and label-list parsing helpers."""
import re


def norm_text(s) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;!\"'`()[]{}<>")


def parse_label_list(raw: str, max_labels: int = 3) -> list:
    """Comma-separated model response -> normalized, deduped labels, in order."""
    parts = [norm_text(p) for p in str(raw).replace("\n", " ").split(",")]
    out = []
    for p in parts:
        if p and p not in out:
            out.append(p)
        if len(out) >= max_labels:
            break
    return out
