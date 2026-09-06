"""Causal uncertainty-threshold policy: a number, or "dynamic"."""


class ThresholdPolicy:
    """threshold="dynamic": no evidence yet for the first batch_size
    images, so retrieval always fires for them. After that, the
    threshold is re-set every batch_size images to the mean uncertainty
    of the batch just completed (not a cumulative mean).

    threshold=<number>: every image compared against that fixed value.
    """

    def __init__(self, threshold, batch_size: int = 10):
        self.is_dynamic = str(threshold).strip().lower() == "dynamic"
        self.fixed = None if self.is_dynamic else float(threshold)
        self.batch_size = batch_size

        self._active = self.fixed
        self._batch_sum = 0.0
        self._batch_count = 0
        self._total_seen = 0

    def current(self) -> float:
        if not self.is_dynamic:
            return self.fixed
        if self._total_seen < self.batch_size:
            return float("-inf")
        return self._active

    def update(self, uncertainty: float) -> None:
        self._total_seen += 1
        if not self.is_dynamic:
            return
        self._batch_sum += uncertainty
        self._batch_count += 1
        if self._batch_count == self.batch_size:
            self._active = self._batch_sum / self._batch_count
            self._batch_sum = 0.0
            self._batch_count = 0
