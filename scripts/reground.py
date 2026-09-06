#!/usr/bin/env python3
"""
Recompute GazeHOI grounding metrics from an already-completed
run_gazehoi_agent.py run, without re-running Qwen3-VL. Reuses the saved
predictions, the saved visual-prompt images, and the saved GazeLLE
heatmaps. Prediction and rescue results are untouched.

Use this after changing anything in source/grounding.py: a different
detector, a fixed selection formula, or a different IoU threshold. The
VLM inference is the expensive part of a GazeHOI run and does not need
repeating just to re-score grounding.

"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from source.config import GazeHOIConfig
from source.data.gazehoi_annotations import load_ground_truth
from source.models.grounding import (
    GROUNDING_FIELDS,
    SELECTION_MODES,
    build_detector,
    print_grounding_summary,
    run_grounding,
)
from source.data.vocab import Vocab

PREDICTION_FIELDS = [
    "file_name", "gt_label",
    "stage1_preds", "stage1_uncertainty", "stage1_top1_correct", "stage1_topn_correct",
    "rag_applied", "rag_preds", "final_pred", "final_correct", "final_topn_correct",
]


def load_gaze_meta(output_dir_of_prompts: str) -> dict:
    path = os.path.join(output_dir_of_prompts, "meta.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {
            row["file_name"]: {"gaze_x_px": float(row["gaze_x_px"]), "gaze_y_px": float(row["gaze_y_px"])}
            for row in csv.DictReader(f)
        }


def load_heatmap(output_dir_of_prompts: str, file_name: str):
    path = os.path.join(output_dir_of_prompts, "heatmaps", file_name + ".npy")
    return np.load(path) if os.path.isfile(path) else None


def is_true(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1")


def split_labels(s: str) -> list:
    return [x for x in str(s).split("|") if x]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-csv", default=None, help="defaults to <output_dir>/gazehoi_results.csv")
    ap.add_argument("--grounding-mode", choices=SELECTION_MODES + ["all"], default=None,
                     help="overrides cfg.grounding_mode")
    args = ap.parse_args()

    cfg = GazeHOIConfig.from_yaml(args.config)
    if args.grounding_mode:
        cfg.grounding_mode = args.grounding_mode
    vocab = Vocab(cfg.vocab_path)

    results_csv = args.results_csv or os.path.join(cfg.output_dir, "gazehoi_results.csv")
    with open(results_csv, newline="", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
    if args.limit:
        pred_rows = pred_rows[: args.limit]
    print(f"[reground] {len(pred_rows)} predictions loaded from {results_csv}")

    gt = load_ground_truth(cfg.eval_csv, cfg.annotations_csv)
    gaze_meta = load_gaze_meta(cfg.query_images_dir)
    detector = build_detector(cfg, vocab)
    print(f"[reground] detector: {cfg.detector}")

    out_rows = []
    for i, pr in enumerate(pred_rows, 1):
        file_name = pr["file_name"]
        row = {k: pr[k] for k in PREDICTION_FIELDS if k in pr}

        entry = gt.get(file_name)
        meta_gaze = gaze_meta.get(file_name)
        image_path = os.path.join(cfg.query_images_dir, file_name)
        if entry is None or meta_gaze is None or not os.path.isfile(image_path):
            out_rows.append(row)
            continue

        top3 = split_labels(pr["rag_preds"]) if is_true(pr["rag_applied"]) and pr["rag_preds"] else split_labels(pr["stage1_preds"])
        image = Image.open(image_path).convert("RGB")
        heatmap = load_heatmap(cfg.query_images_dir, file_name)
        gaze_xy = (meta_gaze["gaze_x_px"], meta_gaze["gaze_y_px"])

        ground_row = run_grounding(detector, image, top3, heatmap, gaze_xy, entry["object_bbox"], grounding_mode=cfg.grounding_mode)
        row.update(ground_row)
        out_rows.append(row)

        if i % 200 == 0 or i == len(pred_rows):
            print(f"[{i:05d}/{len(pred_rows)}] {file_name}")

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS + GROUNDING_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n[reground] updated: {results_csv}")

    print_grounding_summary(out_rows, cfg.iou_threshold)


if __name__ == "__main__":
    main()
