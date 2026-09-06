#!/usr/bin/env python3
"""
for each test instance, sequentially run GazeLLE gaze prediction, 
draw the visual prompt in memory, Qwen3-VL gave top-N prediction,
retrieval rescue if uncertain, then grounding (reusing the gaze 
point and heatmap already computed above).

"""
import argparse
import csv
import dataclasses
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from scripts.build_cls_embeddings import build_embeddings
from source.agent import GazeTargetAgent
from source.config import GazeHOIConfig
from source.data.dataset import GazeHOITestSet
from source.models.gazelle import GazeLLEEstimator
from source.models.grounding import (
    GROUNDING_FIELDS,
    SELECTION_MODES,
    build_detector,
    print_grounding_summary,
    run_grounding,
)
from source.models.model import QwenGazeVLM
from source.models.retrieval import ClsRetriever
from source.threshold import ThresholdPolicy
from source.visual_arrow import draw_gaze_prompt
from source.data.vocab import Vocab

OUTPUT_FIELDS = [
    "file_name", "gt_label",
    "stage1_preds", "stage1_uncertainty", "stage1_top1_correct", "stage1_topn_correct",
    "rag_applied", "rag_preds", "final_pred", "final_correct", "final_topn_correct",
] + GROUNDING_FIELDS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--grounding-mode", choices=SELECTION_MODES + ["all"], default=None,
                     help="overrides cfg.grounding_mode")
    ap.add_argument("--raw-images-dir", default=None, help="overrides cfg.raw_images_dir")
    ap.add_argument("--eval-csv", default=None, help="overrides cfg.eval_csv")
    ap.add_argument("--annotations-csv", default=None, help="overrides cfg.annotations_csv")
    ap.add_argument("--vocab-path", default=None, help="overrides cfg.vocab_path")
    ap.add_argument("--train-annotations-csv", default=None, help="overrides cfg.train_annotations_csv")
    args = ap.parse_args()

    cfg = GazeHOIConfig.from_yaml(args.config)
    if args.grounding_mode:
        cfg.grounding_mode = args.grounding_mode
    if args.raw_images_dir:
        cfg.raw_images_dir = args.raw_images_dir
    if args.eval_csv:
        cfg.eval_csv = args.eval_csv
    if args.annotations_csv:
        cfg.annotations_csv = args.annotations_csv
    if args.vocab_path:
        cfg.vocab_path = args.vocab_path
    if args.train_annotations_csv:
        cfg.train_annotations_csv = args.train_annotations_csv
    random.seed(cfg.seed)

    vocab = Vocab(cfg.vocab_path)
    print(f"[agent] vocab size: {len(vocab)}")

    dataset = GazeHOITestSet(cfg.raw_images_dir, cfg.eval_csv, cfg.annotations_csv)
    items = list(dataset)
    if args.limit:
        items = items[: args.limit]
    print(f"[agent] evaluating {len(items)} GazeHOI instances")

    if not (os.path.isfile(cfg.train_embeds) and os.path.isfile(cfg.train_meta)):
        print("[agent] train CLS embeddings not found, computing them now (one-time, see scripts/build_cls_embeddings.py)")
        build_embeddings(dataclasses.asdict(cfg))

    retriever = ClsRetriever(
        cfg.train_embeds, cfg.train_meta, vocab, label_column=cfg.label_column,
        clip_model=cfg.clip_model, cache_dir=cfg.model_cache_dir, device=cfg.device,
    )
    print(f"[agent] CLS retrieval pool: {len(retriever.train_meta)} labeled train examples")

    vlm = QwenGazeVLM(cfg.qwen_model, device=cfg.device, cache_dir=cfg.model_cache_dir)
    print("[agent] Qwen3-VL loaded")

    estimator = GazeLLEEstimator(
        os.path.join(cfg.model_cache_dir, "torch_hub"), device=cfg.device,
        repo=cfg.gazelle_repo, model_name=cfg.gazelle_model,
    )
    print("[agent] GazeLLE loaded")

    agent = GazeTargetAgent(
        vlm, vocab, retriever,
        top_n=cfg.top_n, rag_top_k=cfg.rag_top_k,
        max_new_tokens_stage1=cfg.max_new_tokens_stage1,
        max_new_tokens_stage2=cfg.max_new_tokens_stage2,
    )
    threshold_policy = ThresholdPolicy(cfg.uncertainty_threshold, cfg.dynamic_batch_size)

    detector = build_detector(cfg, vocab) if cfg.grounding_enabled else None
    if detector is not None:
        print(f"[agent] grounding detector: {cfg.detector}")

    save_debug = cfg.save_visual_prompts or cfg.save_heatmaps
    if save_debug and cfg.query_images_dir:
        os.makedirs(cfg.query_images_dir, exist_ok=True)
        meta_rows = []
        if cfg.save_heatmaps:
            heatmaps_dir = os.path.join(cfg.query_images_dir, "heatmaps")
            os.makedirs(heatmaps_dir, exist_ok=True)

    out_csv = os.path.join(cfg.output_dir, "gazehoi_results.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"\n[run] uncertainty_threshold: {cfg.uncertainty_threshold}\n")

    rows = []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for i, item in enumerate(items, 1):
            image = Image.open(item["raw_image_path"]).convert("RGB")
            pred = estimator.predict(image, item["head_bbox"])
            prompt_image = draw_gaze_prompt(
                image, item["head_bbox"], (pred.x_px, pred.y_px),
                style=cfg.style, color=cfg.color, arrow_width=cfg.arrow_width,
                dot_radius=cfg.dot_radius, heatmap_alpha=cfg.heatmap_alpha, heatmap=pred.heatmap,
            )

            # Predict top-N, then rescue via retrieval if uncertain.
            threshold = threshold_policy.current()
            result = agent.run(
                item["file_name"], prompt_image, item["gt_label"], item["gt_label_set"],
                threshold, retrieval_key=item["retrieval_key"], retrieval_image=image,
            )
            threshold_policy.update(result.stage1_uncertainty)

            row = {
                "file_name": result.file_name,
                "gt_label": result.gt_label,
                "stage1_preds": "|".join(result.stage1_preds),
                "stage1_uncertainty": result.stage1_uncertainty,
                "stage1_top1_correct": result.stage1_top1_correct,
                "stage1_topn_correct": result.stage1_topn_correct,
                "rag_applied": result.rag_applied,
                "rag_preds": "|".join(result.rag_preds) if result.rag_preds else "",
                "final_pred": result.final_pred,
                "final_correct": result.final_correct,
                "final_topn_correct": result.final_topn_correct,
            }

            tag = "RAG" if result.rag_applied else "  -"
            log_line = (
                f"[{i:05d}/{len(items)}] {item['file_name']} | {tag} | "
                f"stage1={result.stage1_preds} | gt='{result.gt_label}' | {'OK' if result.final_correct else 'no'}"
            )

            # Grounding, right after, for the same sample. Reuses the
            # heatmap and gaze point already computed above, with no
            # reload from disk. Runs the detector on the drawn
            # prompt_image, matching what the VLM saw, not the raw image.
            if detector is not None:
                top3 = result.rag_preds if result.rag_applied else result.stage1_preds
                gaze_xy = (pred.x_px, pred.y_px)
                ground_row = run_grounding(detector, prompt_image, top3, pred.heatmap, gaze_xy, item["object_bbox"], grounding_mode=cfg.grounding_mode)
                row.update(ground_row)
                shown = SELECTION_MODES if cfg.grounding_mode == "all" else [cfg.grounding_mode]
                log_line += " | grd(" + ", ".join(f"{m}={ground_row[f'all_{m}_iou']:.2f}" for m in shown) + ")"

            if save_debug and cfg.query_images_dir:
                if cfg.save_visual_prompts:
                    prompt_image.save(os.path.join(cfg.query_images_dir, item["file_name"]), quality=92)
                if cfg.save_heatmaps:
                    hm = pred.heatmap.detach().cpu().numpy() if hasattr(pred.heatmap, "detach") else np.asarray(pred.heatmap)
                    np.save(os.path.join(heatmaps_dir, item["file_name"] + ".npy"), hm.astype(np.float32))
                meta_rows.append({"file_name": item["file_name"], "gaze_x_px": pred.x_px, "gaze_y_px": pred.y_px, "inout_score": pred.inout})

            rows.append(row)
            writer.writerow(row)
            print(log_line)

    if save_debug and cfg.query_images_dir:
        with open(os.path.join(cfg.query_images_dir, "meta.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "gaze_x_px", "gaze_y_px", "inout_score"])
            writer.writeheader()
            writer.writerows(meta_rows)

    print(f"\n[run] results: {out_csv}")

    n = max(len(rows), 1)
    before_top1 = sum(r["stage1_top1_correct"] for r in rows) / n
    before_topn = sum(r["stage1_topn_correct"] for r in rows) / n
    after_top1 = sum(r["final_correct"] for r in rows) / n
    after_topn = sum(r["final_topn_correct"] for r in rows) / n

    print("\n===== PREDICTION SUMMARY =====")
    print(f"Evaluated                    : {len(rows)}")
    print(f"Rescued via retrieval        : {sum(r['rag_applied'] for r in rows)}")
    print()
    print(f"{'Metric':<12} {'Before retrieval':>18} {'After retrieval':>18}")
    print(f"{'Top-1':<12} {before_top1:>18.4f} {after_top1:>18.4f}")
    print(f"{'Top-' + str(cfg.top_n):<12} {before_topn:>18.4f} {after_topn:>18.4f}")

    if detector is not None:
        print_grounding_summary(rows, cfg.iou_threshold)


if __name__ == "__main__":
    main()
