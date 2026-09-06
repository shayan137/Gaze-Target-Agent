"""Vocabulary loading + membership checks for the target-object label set."""
import json

from .text_utils import norm_text


class Vocab:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            vocab2id = json.load(f)
        self.labels = sorted(vocab2id.keys())
        self._norm_set = {norm_text(l) for l in self.labels}

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, label: str) -> bool:
        return norm_text(label) in self._norm_set

    def as_prompt_list(self) -> str:
        return ", ".join(self.labels)
