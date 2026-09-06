#!/usr/bin/env python3
"""
for each test instance, sequentially run GazeLLE gaze prediction, 
draw the visual prompt in memory, Qwen3-VL gave top-N prediction,
retrieval rescue if uncertain
"""
import argparse
import csv
import dataclasses
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

from scripts.build_cls_embeddings import build_embeddings
from source.agent import GazeTargetAgent
from source.config import AgentConfig
from source.data.dataset import GazeFollowTestSet
from source.models.gazelle import GazeLLEEstimator
from source.data.io_utils import write_results_csv
from source.models.model import QwenGazeVLM
from source.models.retrieval import ClsRetriever
from source.threshold import ThresholdPolicy
from source.visual_arrow import draw_gaze_prompt
from source.data.vocab import Vocab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml file")
    ap.add_argument("--limit", type=int, default=None, help="only run on the first N images (debug)")
    ap.add_argument("--raw-images-dir", default=None, help="overrides cfg.raw_images_dir")
    ap.add_argument("--annotations-txt", default=None, help="overrides cfg.annotations_txt")
    ap.add_argument("--gt-csv", default=None, help="overrides cfg.gt_csv")
    ap.add_argument("--vocab-path", default=None, help="overrides cfg.vocab_path")
    ap.add_argument("--train-labels-csv", default=None, help="overrides cfg.train_labels_csv")
    args = ap.parse_args()

    cfg = AgentConfig.from_yaml(args.config)
    if args.raw_images_dir:
        cfg.raw_images_dir = args.raw_images_dir
    if args.annotations_txt:
        cfg.annotations_txt = args.annotations_txt
    if args.gt_csv:
        cfg.gt_csv = args.gt_csv
    if args.vocab_path:
        cfg.vocab_path = args.vocab_path
    if args.train_labels_csv:
        cfg.train_labels_csv = args.train_labels_csv
    random.seed(cfg.seed)

    vocab = Vocab(cfg.vocab_path)
    print(f"[agent] vocab size: {len(vocab)}")

    dataset = GazeFollowTestSet(cfg.raw_images_dir, cfg.annotations_txt, cfg.gt_csv)
    items = list(dataset)
    if args.limit:
        items = items[: args.limit]
    print(f"[agent] evaluating {len(items)} images")

    if not (os.path.isfile(cfg.train_embeds) and os.path.isfile(cfg.train_meta)):
        print("[agent] train CLS embeddings not found, computing them now (one-time, see scripts/build_cls_embeddings.py)")
        build_embeddings(dataclasses.asdict(cfg))

    retriever = ClsRetriever(
        cfg.train_embeds, cfg.train_meta, vocab,
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

    if cfg.save_visual_prompts and cfg.query_images_dir:
        os.makedirs(cfg.query_images_dir, exist_ok=True)
        meta_rows = []

    out_csv = os.path.join(cfg.output_dir, "gazefollow_results.csv")
    print(f"\n[run] uncertainty_threshold: {cfg.uncertainty_threshold}\n")

    results = []
    for i, item in enumerate(items, 1):
        image = Image.open(item["raw_image_path"]).convert("RGB")
        pred = estimator.predict(image, item["head_bbox"])
        prompt_image = draw_gaze_prompt(
            image, item["head_bbox"], (pred.x_px, pred.y_px),
            style=cfg.style, color=cfg.color, arrow_width=cfg.arrow_width,
            dot_radius=cfg.dot_radius, heatmap_alpha=cfg.heatmap_alpha, heatmap=pred.heatmap,
        )

        threshold = threshold_policy.current()
        result = agent.run(
            item["file_name"], prompt_image, item["gt_label"], item["gt_label_set"],
            threshold, retrieval_image=image,
        )
        threshold_policy.update(result.stage1_uncertainty)
        results.append(result)

        if cfg.save_visual_prompts and cfg.query_images_dir:
            prompt_image.save(os.path.join(cfg.query_images_dir, item["file_name"]), quality=92)
            meta_rows.append({"file_name": item["file_name"], "gaze_x_px": pred.x_px, "gaze_y_px": pred.y_px, "inout_score": pred.inout})

        tag = "RAG" if result.rag_applied else "  -"
        print(
            f"[{i:05d}/{len(items)}] {item['file_name']} | {tag} | thr={threshold:.3f} | "
            f"stage1={result.stage1_preds} (unc={result.stage1_uncertainty:.3f})"
            + (f" -> rag={result.rag_preds}" if result.rag_applied else "")
            + f" | gt='{result.gt_label}' | {'OK' if result.final_correct else 'no'}"
        )

        if i % 50 == 0:
            write_results_csv(out_csv, results)

    write_results_csv(out_csv, results)

    if cfg.save_visual_prompts and cfg.query_images_dir:
        with open(os.path.join(cfg.query_images_dir, "meta.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "gaze_x_px", "gaze_y_px", "inout_score"])
            writer.writeheader()
            writer.writerows(meta_rows)

    # "before" is the first prediction alone. "after" is the final prediction: the rescue result where it fired, otherwise the first.
    n = max(len(results), 1)
    before_top1 = sum(r.stage1_top1_correct for r in results) / n
    before_topn = sum(r.stage1_topn_correct for r in results) / n
    before_multi = sum(r.stage1_multi_correct for r in results) / n
    after_top1 = sum(r.final_correct for r in results) / n
    after_topn = sum(r.final_topn_correct for r in results) / n
    after_multi = sum(r.final_multi_correct for r in results) / n

    rescued = [r for r in results if r.rag_applied]

    print("\n===== SUMMARY =====")
    print(f"Evaluated                    : {len(results)}")
    print(f"Rescued via retrieval        : {len(rescued)} ({len(rescued) / n:.1%})")
    print()
    print(f"{'Metric':<12} {'Before retrieval':>18} {'After retrieval':>18}")
    print(f"{'Top-1':<12} {before_top1:>18.4f} {after_top1:>18.4f}")
    print(f"{'Top-' + str(cfg.top_n):<12} {before_topn:>18.4f} {after_topn:>18.4f}")
    print(f"{'Multi-acc':<12} {before_multi:>18.4f} {after_multi:>18.4f}")
    print(f"\nResults                      : {out_csv}")


if __name__ == "__main__":
    main()
