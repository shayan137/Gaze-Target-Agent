"""CLS-embedding retrieval for the rescue step: nearest training neighbors
by cosine similarity, one per distinct label. Only labels are used
downstream, no images, so no image paths are resolved here.

The query image's embedding is computed live, not precomputed, using
the same CLIP encoder that built the training embeddings. See
scripts/build_cls_embeddings.py."""
import os

import numpy as np
import pandas as pd

from ..data.text_utils import norm_text


class ClsRetriever:
    def __init__(
        self,
        train_embeds_path: str,
        train_meta_path: str,
        vocab,
        label_column: str = "gt_label",
        clip_model: str = "openai/clip-vit-base-patch32",
        cache_dir: str = None,
        device: str = "cuda",
    ):
        train_embeds = np.load(train_embeds_path).astype(np.float32)
        train_meta = pd.read_csv(train_meta_path)
        self.label_column = label_column

        gt_norm = train_meta[label_column].astype(str).map(norm_text)
        valid_gt = gt_norm.notna() & (gt_norm != "") & (gt_norm != "nan")
        in_vocab = gt_norm.isin({norm_text(l) for l in vocab.labels})
        keep = (valid_gt & in_vocab).to_numpy()

        self.train_meta = train_meta.loc[keep].reset_index(drop=True)
        self.train_embeds = train_embeds[keep]

        from transformers import CLIPModel, CLIPProcessor

        hf_cache = os.path.join(cache_dir, "huggingface") if cache_dir else None
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=hf_cache)
        self.model = CLIPModel.from_pretrained(clip_model, cache_dir=hf_cache).to(device).eval()
        self._embed_cache = {}

    def embed_image(self, image) -> np.ndarray:
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            # Newer transformers returns a BaseModelOutputWithPooling
            # here, not a plain tensor. The projected embedding is in
            # its pooler_output field.
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].detach().cpu().numpy().astype(np.float32)

    def top_k_unique_gt(self, image, k: int = 6, cache_key: str = None):
        """Up to k (meta_row, similarity) pairs, most similar first, one
        per distinct label. `cache_key` avoids re-embedding the same raw
        photo for multiple GazeHOI instances that share it."""
        if cache_key is not None and cache_key in self._embed_cache:
            q_vec = self._embed_cache[cache_key]
        else:
            q_vec = self.embed_image(image)
            if cache_key is not None:
                self._embed_cache[cache_key] = q_vec

        sims = self.train_embeds @ q_vec  # both sides are L2-normalized, so this is cosine similarity
        order = np.argsort(-sims)

        chosen, seen_labels = [], set()
        for idx in order:
            row = self.train_meta.iloc[int(idx)]
            label = norm_text(row[self.label_column])
            if label in seen_labels:
                continue
            seen_labels.add(label)
            # agent.py always reads row["gt_label"], regardless of the
            # dataset's own column name (GazeHOI's is "gt_object"), so
            # alias it here.
            row = row.copy()
            row["gt_label"] = row[self.label_column]
            chosen.append((row, float(sims[int(idx)])))
            if len(chosen) >= k:
                break
        return chosen
