"""GazeTargetAgent: predict top-N, check confidence, and retrieve plus
re-predict if it's low. All in one call per image, with no separate
second pass over the dataset."""
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

from .data.text_utils import norm_text, parse_label_list
from .prompts import build_rag_prompt, build_topk_prompt
from .uncertainty import first_token_entropy, sequence_probability


@dataclass
class GazeTargetResult:
    file_name: str
    gt_label: str
    gt_label_set: List[str] = field(default_factory=list)  # every valid label, for multi-acc

    stage1_preds: List[str] = field(default_factory=list)
    stage1_pred_prob: float = 0.0
    stage1_uncertainty: float = 0.0
    stage1_entropy: float = float("nan")
    stage1_top1_correct: bool = False
    stage1_topn_correct: bool = False
    stage1_multi_correct: bool = False  # top-1 matches ANY valid label

    rag_applied: bool = False
    rag_candidates: Optional[List[str]] = None  # retrieved-label pool shown to the model
    rag_preds: Optional[List[str]] = None        # the 3 labels the model picked this round
    rag_top1_correct: Optional[bool] = None
    rag_topn_correct: Optional[bool] = None
    rag_multi_correct: Optional[bool] = None

    final_pred: str = ""
    final_correct: bool = False        # top-1, after retrieval if it ran
    final_topn_correct: bool = False   # top-3, after retrieval if it ran
    final_multi_correct: bool = False  # multi-acc, after retrieval if it ran


class GazeTargetAgent:
    def __init__(
        self,
        vlm,
        vocab,
        retriever,
        top_n: int = 3,
        rag_top_k: int = 6,
        max_new_tokens_stage1: int = 32,
        max_new_tokens_stage2: int = 32,
    ):
        self.vlm = vlm
        self.vocab = vocab
        self.retriever = retriever
        self.top_n = top_n
        self.rag_top_k = rag_top_k
        self.max_new_tokens_stage1 = max_new_tokens_stage1
        self.max_new_tokens_stage2 = max_new_tokens_stage2
        self._vocab_prompt_str = vocab.as_prompt_list()

    # one image in, one final result out
    def run(
        self,
        file_name: str,
        image: Image.Image,
        gt_label: str,
        gt_label_set: List[str],
        uncertainty_threshold: float,
        retrieval_key: Optional[str] = None,
        retrieval_image: Optional[Image.Image] = None,
    ) -> GazeTargetResult:
        # image is the drawn visual-prompt image, shown to the VLM in
        # both stages. retrieval_image is the raw, un-annotated photo,
        # used for CLS retrieval instead, since training embeddings are
        # of raw scenes. It defaults to image if not given.
        # retrieval_key is an embedding-cache key, used when it differs
        # from file_name. For example GazeHOI keys by the raw photo
        # filename, since one photo can host several test instances and
        # should not be re-embedded for each one.
        retrieval_key = retrieval_key or file_name
        retrieval_image = image if retrieval_image is None else retrieval_image
        result = self.predict_stage1(file_name, image, gt_label, gt_label_set)
        if result.stage1_uncertainty > uncertainty_threshold:
            self.rescue_stage2(result, image, retrieval_image, retrieval_key)
        return result

    # step 1: predict top-N in a single generation call
    def predict_stage1(
        self, file_name: str, image: Image.Image, gt_label: str, gt_label_set: List[str]
    ) -> GazeTargetResult:
        prompt = build_topk_prompt(self._vocab_prompt_str, k=self.top_n)

        text, scores, gen_ids = self.vlm.generate(
            image, prompt, max_new_tokens=self.max_new_tokens_stage1, with_scores=True
        )

        preds = parse_label_list(text, max_labels=self.top_n)
        in_vocab_preds = [p for p in preds if p in self.vocab]
        preds = in_vocab_preds or preds[:1]

        prob = sequence_probability(scores, gen_ids)
        uncertainty = 1.0 - prob
        entropy = first_token_entropy(scores)

        top1 = preds[0] if preds else ""
        top1_correct = top1 == gt_label
        topn_correct = gt_label in set(preds)
        multi_correct = top1 in set(gt_label_set)

        return GazeTargetResult(
            file_name=file_name,
            gt_label=gt_label,
            gt_label_set=gt_label_set,
            stage1_preds=preds,
            stage1_pred_prob=prob,
            stage1_uncertainty=uncertainty,
            stage1_entropy=entropy,
            stage1_top1_correct=top1_correct,
            stage1_topn_correct=topn_correct,
            stage1_multi_correct=multi_correct,
            final_pred=top1,
            final_correct=top1_correct,
            final_topn_correct=topn_correct,
            final_multi_correct=multi_correct,
        )

    # step 3: CLS retrieval (labels only) + restricted top-3 re-prediction
    def rescue_stage2(
        self, result: GazeTargetResult, image: Image.Image, retrieval_image: Image.Image, retrieval_key: str
    ) -> GazeTargetResult:
        neighbors = self.retriever.top_k_unique_gt(retrieval_image, k=self.rag_top_k, cache_key=retrieval_key)
        if not neighbors:
            return result

        retrieved_labels = [norm_text(row["gt_label"]) for row, _ in neighbors]
        candidate_pool = list(dict.fromkeys(retrieved_labels + result.stage1_preds))

        prompt = build_rag_prompt(retrieved_labels, result.stage1_preds, top_k=self.rag_top_k)
        text, _, _ = self.vlm.generate(
            image,
            prompt,
            max_new_tokens=self.max_new_tokens_stage2,
            with_scores=False,
        )

        preds = parse_label_list(text, max_labels=self.top_n)
        in_pool_preds = [p for p in preds if p in candidate_pool]
        preds = in_pool_preds or preds[:1]

        top1 = preds[0] if preds else ""
        result.rag_applied = True
        result.rag_candidates = candidate_pool
        result.rag_preds = preds
        result.rag_top1_correct = top1 == result.gt_label
        result.rag_topn_correct = result.gt_label in set(preds)
        result.rag_multi_correct = top1 in set(result.gt_label_set)
        result.final_pred = top1
        result.final_correct = result.rag_top1_correct
        result.final_topn_correct = result.rag_topn_correct
        result.final_multi_correct = result.rag_multi_correct
        return result
