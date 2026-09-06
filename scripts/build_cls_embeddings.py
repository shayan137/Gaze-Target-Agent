#!/usr/bin/env python3
"""
Precompute CLIP CLS embeddings for a dataset's training images. This is
the retrieval pool ClsRetriever uses for the RAG rescue step. Run once
per dataset before the first full pipeline run. The main pipeline
scripts just load the resulting train_embeds.npy/train_meta.csv. Query
image embeddings are never precomputed, they are computed live at
rescue time against this pool (see source/retrieval.py).

"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

BATCH_SIZE = 32


def gazefollow_train_items(cfg):
    df = pd.read_csv(cfg["train_labels_csv"])
    df = df[df["gaze_pseudo_label"].notna()].copy()
    for _, row in df.iterrows():
        rel = str(row["path"])  # e.g. "train/00000000/00000001.jpg"
        yield {
            "file_name": os.path.basename(rel),
            "rel_path": rel,
            "img_path": os.path.join(cfg["raw_images_dir"], rel),
            "gt_label": row["gaze_pseudo_label"],
        }


def gazehoi_train_items(cfg):
    df = pd.read_csv(cfg["train_annotations_csv"])
    for _, row in df.iterrows():
        yield {
            "file_name": row["file_name"],
            "img_path": os.path.join(cfg["raw_images_dir"], row["file_name"]),
            "gt_object": row["object"],
        }


DATASET_LOADERS = {"gazefollow": gazefollow_train_items, "gazehoi": gazehoi_train_items}
META_FIELDS = {
    "gazefollow": ["file_name", "rel_path", "img_path", "gt_label"],
    "gazehoi": ["file_name", "img_path", "gt_object"],
}


@torch.no_grad()
def embed_batch(paths, processor, model, device):
    images, kept = [], []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            kept.append(p)
        except Exception:
            continue
    if not images:
        return {}
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    # Newer transformers returns a BaseModelOutputWithPooling here, not
    # a plain tensor. The projected embedding is in pooler_output.
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    feats = torch.nn.functional.normalize(feats, dim=-1)
    feats = feats.detach().cpu().numpy().astype(np.float32)
    return dict(zip(kept, feats))


def build_embeddings(cfg: dict, limit: int = None) -> None:
    """cfg is a plain dict with at least dataset, raw_images_dir,
    train_*_csv, train_embeds, train_meta, model_cache_dir, clip_model,
    and device (a dataclass works too, via dataclasses.asdict(cfg)).
    run_agent.py and run_gazehoi_agent.py call this automatically if the
    embeddings do not exist yet, so it rarely needs running by hand."""
    items = list(DATASET_LOADERS[cfg["dataset"]](cfg))
    if limit:
        items = items[:limit]
    print(f"[embed] dataset={cfg['dataset']} annotation rows={len(items)}")

    from transformers import CLIPModel, CLIPProcessor

    cache_dir = cfg.get("model_cache_dir")
    hf_cache = os.path.join(cache_dir, "huggingface") if cache_dir else None
    clip_name = cfg.get("clip_model", "openai/clip-vit-base-patch32")
    device = cfg.get("device", "cuda")
    processor = CLIPProcessor.from_pretrained(clip_name, cache_dir=hf_cache)
    model = CLIPModel.from_pretrained(clip_name, cache_dir=hf_cache).to(device).eval()
    print(f"[embed] CLIP loaded: {clip_name}")

    # Embed each unique image once, then reuse that vector for every
    # meta row that references it. GazeHOI has about 9k images that
    # appear in multiple annotation rows, so this avoids embedding the
    # same picture more than once.
    unique_paths = sorted({it["img_path"] for it in items})
    print(f"[embed] {len(unique_paths)} unique images to embed")

    vec_by_path, bad = {}, 0
    for start in range(0, len(unique_paths), BATCH_SIZE):
        batch = unique_paths[start : start + BATCH_SIZE]
        result = embed_batch(batch, processor, model, device)
        bad += len(batch) - len(result)
        vec_by_path.update(result)
        done = min(start + BATCH_SIZE, len(unique_paths))
        if (start // BATCH_SIZE) % 20 == 0 or done == len(unique_paths):
            print(f"[embed] [{done:06d}/{len(unique_paths)}] embedded, {bad} unreadable so far")

    out_rows, out_vecs = [], []
    for it in items:
        vec = vec_by_path.get(it["img_path"])
        if vec is None:
            continue
        out_rows.append({k: it[k] for k in META_FIELDS[cfg["dataset"]]})
        out_vecs.append(vec)

    embeds = np.stack(out_vecs, axis=0).astype(np.float32)
    meta = pd.DataFrame(out_rows)

    out_embeds, out_meta = cfg["train_embeds"], cfg["train_meta"]
    os.makedirs(os.path.dirname(out_embeds), exist_ok=True)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    np.save(out_embeds, embeds)
    meta.to_csv(out_meta, index=False)

    print(f"\n[embed] wrote {embeds.shape[0]} meta rows ({len(unique_paths) - bad}/{len(unique_paths)} unique images embedded, {bad} unreadable)")
    print(f"[embed] embeddings: {out_embeds} {embeds.shape}")
    print(f"[embed] meta      : {out_meta}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    build_embeddings(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
