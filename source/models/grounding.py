"""Grounding stage (GazeHOI only): pick one detector box for the gaze
target, then score it against the ground truth box.

Three selection modes, set via GazeHOIConfig.grounding_mode:
- "gaze_point": the candidate box containing GazeLLE's peak gaze pixel.
- "heatmap": among boxes containing the gaze pixel, the one with the
  highest mean heatmap density.
- "sqrt_norm_heatmap": same pool as "heatmap", ranked by heatmap mass
  divided by sqrt(area) instead of mean density. This is the
  best-performing method (see select_sqrt_norm_heatmap).

"Upper Bound" is not a selection method. It is the best IoU achievable
among the candidates, reported as a reference ceiling.
"""
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Detection:
    box_xyxy: tuple  # (x1, y1, x2, y2) pixel coords
    class_name: str
    score: float


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def upper_bound(detections: List[Detection], gt_box):
    """The candidate with the best IoU against gt_box. Not a real
    selection, just a reference ceiling. Returns (Detection, iou), or
    (None, 0.0) if there are no candidates."""
    if not detections:
        return None, 0.0
    best = max(detections, key=lambda d: compute_iou(d.box_xyxy, gt_box))
    return best, compute_iou(best.box_xyxy, gt_box)


def average_precision(scores: List[float], hits: List[bool]) -> float:
    """Standard interpolated AP (PASCAL VOC / COCO style) for one class."""
    import numpy as np

    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    labels = np.asarray(hits, dtype=np.float64)[order]
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0

    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / (tp + fp)
    recall = tp / n_pos

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap, prev_recall = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def mean_average_precision(scores: List[float], hits: List[bool], class_labels: List[str]) -> float:
    """AP@50: compute AP per ground-truth class, then average across
    classes. This matches how detection benchmarks usually report AP.
    mIoU is not grouped by class, only this is."""
    import numpy as np
    from collections import defaultdict

    by_class = defaultdict(lambda: ([], []))
    for score, hit, label in zip(scores, hits, class_labels):
        by_class[label][0].append(score)
        by_class[label][1].append(hit)

    class_aps = [average_precision(s, h) for s, h in by_class.values()]
    return float(np.mean(class_aps)) if class_aps else 0.0


def restrict_to_top3_classes(detections: List[Detection], top3_labels: List[str]) -> List[Detection]:
    labels = {l.strip().lower() for l in top3_labels}
    return [d for d in detections if d.class_name.strip().lower() in labels]


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def contains(box, point):
    x1, y1, x2, y2 = box
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def gaze_point_fallback(detections: List[Detection], gaze_xy) -> Detection:
    """Used when no candidate box contains the gaze point. Picks the box
    whose center is closest to the gaze point."""
    gx, gy = gaze_xy
    return min(detections, key=lambda d: math.hypot(box_center(d.box_xyxy)[0] - gx, box_center(d.box_xyxy)[1] - gy))


def select_gaze_point(detections: List[Detection], gaze_xy) -> Optional[Detection]:
    """Picks the smallest box that contains the gaze point. Falls back
    to the nearest box center if no box contains it."""
    if not detections:
        return None
    containing = [d for d in detections if contains(d.box_xyxy, gaze_xy)]
    if containing:
        return min(containing, key=lambda d: box_area(d.box_xyxy))
    return gaze_point_fallback(detections, gaze_xy)


def containing_candidates(detections: List[Detection], gaze_xy) -> List[Detection]:
    return [d for d in detections if contains(d.box_xyxy, gaze_xy)]


def resize_heatmap(heatmap, image_size):
    """Upsamples GazeLLE's raw heatmap grid to full image size, using
    float32 bilinear interpolation for precision."""
    import numpy as np
    import torch

    hm = heatmap.detach().cpu().numpy() if hasattr(heatmap, "detach") else np.asarray(heatmap)
    W, H = image_size
    t = torch.as_tensor(hm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def heatmap_box_stat(heatmap_full, box, W, H, agg: str) -> float:
    x1, y1, x2, y2 = box
    x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
    x2i, y2i = min(W, int(round(x2))), min(H, int(round(y2)))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    region = heatmap_full[y1i:y2i, x1i:x2i]
    if agg == "mean":
        return float(region.mean())
    if agg == "sqrt_norm":
        # Sum divided by sqrt(area). A box needs real heat mass to win,
        # but is not punished as heavily for being large as a plain mean
        # would punish it.
        area = region.size
        return float(region.sum()) / math.sqrt(area) if area > 0 else 0.0
    raise ValueError(f"unknown heatmap aggregation: {agg!r} (expected mean or sqrt_norm)")


def heatmap_scores(detections: List[Detection], heatmap, image_size, agg: str):
    """Returns (Detection, score) pairs for one heatmap aggregation."""
    hm_full = resize_heatmap(heatmap, image_size)
    W, H = image_size
    return [(d, heatmap_box_stat(hm_full, d.box_xyxy, W, H, agg)) for d in detections]


def select_heatmap(detections: List[Detection], heatmap, image_size, gaze_xy):
    """Among boxes containing the gaze point, picks the one with the
    highest mean heatmap density. Falls back to the nearest box center
    if no box contains the point."""
    if not detections:
        return None, 0.0
    containing = containing_candidates(detections, gaze_xy)
    if containing:
        return max(heatmap_scores(containing, heatmap, image_size, "mean"), key=lambda t: t[1])
    fallback = gaze_point_fallback(detections, gaze_xy)
    return fallback, fallback.score


def select_sqrt_norm_heatmap(detections: List[Detection], heatmap, image_size, gaze_xy):
    """Same candidate pool as select_heatmap, but ranked by heatmap mass
    divided by sqrt(area) instead of mean density. This is the
    best-performing method tested."""
    if not detections:
        return None, 0.0
    containing = containing_candidates(detections, gaze_xy)
    if containing:
        return max(heatmap_scores(containing, heatmap, image_size, "sqrt_norm"), key=lambda t: t[1])
    fallback = gaze_point_fallback(detections, gaze_xy)
    return fallback, fallback.score


class RFDETRDetector:
    """RF-DETR 2X-Large object detector, closed-set (COCO's 80 classes).
    Auto-downloads its checkpoint on first use."""

    def __init__(self, variant: str = "2xlarge", confidence_threshold: float = 0.5, cache_dir: str = None):
        self.threshold = confidence_threshold
        if cache_dir:
            import os

            # Same idea as gazelle.py's TORCH_HOME: tell the library
            # where to cache its downloaded weights.
            os.environ.setdefault("RF_HOME", os.path.join(cache_dir, "rfdetr"))
        if variant == "2xlarge":
            from rfdetr import RFDETR2XLarge as ModelCls
        elif variant == "seg2xlarge":
            from rfdetr import RFDETRSeg2XLarge as ModelCls
        else:
            raise ValueError(f"unknown rfdetr variant: {variant!r} (expected 2xlarge or seg2xlarge)")
        self.model = ModelCls()

    def detect(self, image) -> List[Detection]:
        det = self.model.predict(image, threshold=self.threshold)
        # Read class names from the library's own output instead of
        # indexing a class list by id ourselves. COCO class ids are not
        # contiguous, so indexing by id directly gives the wrong name.
        names = det.data["class_name"]
        return [
            Detection(box_xyxy=tuple(float(v) for v in box), class_name=str(name), score=float(score))
            for box, name, score in zip(det.xyxy, names, det.confidence)
        ]


class Yolo26Detector:
    """Closed-set (COCO-pretrained by default) YOLO26 detector."""

    def __init__(self, weights: str = "yolo26n.pt", device: str = "cuda", confidence_threshold: float = 0.5, cache_dir: str = None):
        import os

        from ultralytics import YOLO

        # A bare filename with no folder resolves under cache_dir/yolo26.
        # A full path in the config still overrides this.
        if cache_dir and not os.path.dirname(weights):
            weights_dir = os.path.join(cache_dir, "yolo26")
            os.makedirs(weights_dir, exist_ok=True)
            weights = os.path.join(weights_dir, weights)

        self.model = YOLO(weights)
        self.device = device
        self.threshold = confidence_threshold

    def detect(self, image) -> List[Detection]:
        import numpy as np

        result = self.model.predict(np.array(image), device=self.device, conf=self.threshold, verbose=False)[0]
        names = result.names
        return [
            Detection(box_xyxy=tuple(float(v) for v in box), class_name=names[int(cls_id)], score=float(score))
            for box, cls_id, score in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
            )
        ]


class SAM3Detector:
    """Open-vocabulary detector, queried with the full label vocabulary
    each call. Not RF-DETR's independently verified level of testing.
    A documented fallback option, RF-DETR is the main detector."""

    def __init__(self, vocab: List[str], checkpoint: str = "facebook/sam3", device: str = "cuda", confidence_threshold: float = 0.5, cache_dir: str = None):
        import os

        from transformers import Sam3Model, Sam3Processor
        import torch

        hf_cache = os.path.join(cache_dir, "huggingface") if cache_dir else None
        self.torch = torch
        self.device = device
        self.threshold = confidence_threshold
        self.vocab = vocab
        self.processor = Sam3Processor.from_pretrained(checkpoint, cache_dir=hf_cache)
        self.model = Sam3Model.from_pretrained(checkpoint, cache_dir=hf_cache).eval().to(device)

    def detect(self, image) -> List[Detection]:
        inputs = self.processor(images=image, text=self.vocab, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([image.size[::-1]])
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]

        return [
            Detection(
                box_xyxy=tuple(float(v) for v in box.tolist()),
                class_name=str(label),
                score=float(score),
            )
            for box, label, score in zip(results["boxes"], results["labels"], results["scores"])
        ]


DETECTORS = {"rfdetr": RFDETRDetector, "yolo26": Yolo26Detector, "sam3": SAM3Detector}


def build_detector(cfg, vocab):
    """Builds the detector named in cfg.detector, caching its weights
    under cfg.model_cache_dir."""
    kind = cfg.detector
    cache_dir = getattr(cfg, "model_cache_dir", None)
    if kind == "rfdetr":
        return DETECTORS["rfdetr"](variant=cfg.rfdetr_variant, confidence_threshold=cfg.detector_confidence_threshold, cache_dir=cache_dir)
    if kind == "yolo26":
        return DETECTORS["yolo26"](weights=cfg.yolo26_weights, confidence_threshold=cfg.detector_confidence_threshold, cache_dir=cache_dir)
    if kind == "sam3":
        return DETECTORS["sam3"](vocab=vocab.labels, checkpoint=cfg.sam3_checkpoint, confidence_threshold=cfg.detector_confidence_threshold, cache_dir=cache_dir)
    raise ValueError(f"unknown detector: {kind!r} (expected rfdetr, yolo26, or sam3)")


CANDIDATE_SETTINGS = ["all", "top3_classes"]
SELECTION_MODES = ["gaze_point", "heatmap", "sqrt_norm_heatmap"]
MODE_LABELS = {
    "gaze_point": "Gaze-Point (smallest, center-fallback)",
    "heatmap": "Heatmap (containing, mean density)",
    "sqrt_norm_heatmap": "Sqrt-normalized heatmap aggregation",
}

GROUNDING_FIELDS = []
for _setting in CANDIDATE_SETTINGS:
    GROUNDING_FIELDS += [f"{_setting}_num_candidates", f"{_setting}_upper_bound_iou", f"{_setting}_upper_bound_score"]
    for _mode in SELECTION_MODES:
        GROUNDING_FIELDS += [f"{_setting}_{_mode}_iou", f"{_setting}_{_mode}_score"]


def run_grounding(detector, image, top3_labels, heatmap, gaze_xy, gt_box, grounding_mode: str = "all") -> dict:
    """Runs the detector once and scores the requested selection mode(s)
    against gt_box, for both candidate pools (all detections, and
    detections restricted to the top-3 predicted classes)."""
    detections_all = detector.detect(image)
    pools = {"all": detections_all, "top3_classes": restrict_to_top3_classes(detections_all, top3_labels)}
    modes = SELECTION_MODES if grounding_mode == "all" else [grounding_mode]

    row = {}
    for setting, candidates in pools.items():
        row[f"{setting}_num_candidates"] = len(candidates)

        ub_det, ub_iou = upper_bound(candidates, gt_box)
        row[f"{setting}_upper_bound_iou"] = ub_iou
        row[f"{setting}_upper_bound_score"] = ub_det.score if ub_det else 0.0

        if "gaze_point" in modes:
            gp = select_gaze_point(candidates, gaze_xy)
            row[f"{setting}_gaze_point_iou"] = compute_iou(gp.box_xyxy, gt_box) if gp else 0.0
            # Rank by detector confidence, since containment is a yes/no
            # decision with no confidence value of its own.
            row[f"{setting}_gaze_point_score"] = gp.score if gp else 0.0

        if "heatmap" in modes:
            hm, hm_score = select_heatmap(candidates, heatmap, image.size, gaze_xy) if heatmap is not None else (None, 0.0)
            row[f"{setting}_heatmap_iou"] = compute_iou(hm.box_xyxy, gt_box) if hm else 0.0
            row[f"{setting}_heatmap_score"] = hm_score

        if "sqrt_norm_heatmap" in modes:
            sh, sh_score = select_sqrt_norm_heatmap(candidates, heatmap, image.size, gaze_xy) if heatmap is not None else (None, 0.0)
            row[f"{setting}_sqrt_norm_heatmap_iou"] = compute_iou(sh.box_xyxy, gt_box) if sh else 0.0
            row[f"{setting}_sqrt_norm_heatmap_score"] = sh_score
    return row


def print_grounding_summary(rows: List[dict], iou_threshold: float) -> None:
    import numpy as np

    grounded = [r for r in rows if "all_upper_bound_iou" in r]
    skipped = len(rows) - len(grounded)
    print("\n===== GROUNDING SUMMARY =====")
    if skipped:
        print(f"(skipped {skipped}/{len(rows)} instances with no gaze-meta entry, e.g. an interrupted run)")
    rows = grounded
    if not rows:
        return
    class_labels = [r["gt_label"] for r in rows]
    header = f"{'Detector method':<22} {'mIoU':>8} {'R@50':>8} {'AP@50':>8}"
    modes_present = [m for m in SELECTION_MODES if f"all_{m}_iou" in rows[0]]

    for setting_label, setting in [("All Detections", "all"), ("Top-3 Classes", "top3_classes")]:
        print(f"\n{setting_label}")
        print(header)
        ious_ub = [r[f"{setting}_upper_bound_iou"] for r in rows]
        scores_ub = [r[f"{setting}_upper_bound_score"] for r in rows]
        hits_ub = [i >= iou_threshold for i in ious_ub]
        ap_ub = mean_average_precision(scores_ub, hits_ub, class_labels)
        print(f"{'Upper Bound':<22} {np.mean(ious_ub):>8.4f} {np.mean(hits_ub):>8.4f} {ap_ub:>8.4f}")
        for mode in modes_present:
            ious = [r[f"{setting}_{mode}_iou"] for r in rows]
            scores = [r[f"{setting}_{mode}_score"] for r in rows]
            hits = [i >= iou_threshold for i in ious]
            ap = mean_average_precision(scores, hits, class_labels)
            print(f"{MODE_LABELS[mode]:<22} {np.mean(ious):>8.4f} {np.mean(hits):>8.4f} {ap:>8.4f}")
