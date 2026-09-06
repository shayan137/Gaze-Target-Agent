"""Test-split loaders: pair each raw image with its head bbox (for
GazeLLE) and its ground-truth target-object label (GazeFollow), or with
its label plus retrieval key, head bbox, and grounding GT box (GazeHOI).
Driven entirely by the annotation and ground-truth files. No pre-built
visual-prompt directory is needed."""
import os

import pandas as pd

from .gazefollow_annotations import load_head_bboxes
from .gazehoi_annotations import load_ground_truth
from .text_utils import norm_text


class GazeFollowTestSet:
    def __init__(self, raw_images_dir: str, annotations_txt: str, gt_csv: str):
        self.raw_images_dir = raw_images_dir
        heads = load_head_bboxes(annotations_txt)

        df = pd.read_csv(gt_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        # A missing gaze_gt_label means no ground truth. Drop
        # these rows, otherwise str(NaN) becomes the literal label "nan".
        df = df[df["gaze_gt_label"].notna()].copy()
        df["filename"] = df["path"].apply(lambda p: os.path.basename(str(p)))
        df["gt_norm"] = df["gaze_gt_label"].apply(norm_text)
        gt = dict(zip(df["filename"], df["gt_norm"]))

        # gaze_gt_labels (plural) is hyphen-separated, e.g. "bowl-food",
        # for images with more than one valid label. gaze_gt_label is just
        # the first. Used for multi-acc: credit a prediction matching any.
        df["gt_label_set"] = df["gaze_gt_labels"].apply(
            lambda s: [norm_text(x) for x in str(s).split("-")]
        )
        gt_multi = dict(zip(df["filename"], df["gt_label_set"]))

        self.items = []
        for file_name, entry in sorted(heads.items()):
            if file_name not in gt:
                continue
            raw_image_path = os.path.join(raw_images_dir, entry["rel_path"])
            if not os.path.isfile(raw_image_path):
                continue
            self.items.append({
                "file_name": file_name,
                "raw_image_path": raw_image_path,
                "head_bbox": entry["head_bbox"],
                "gt_label": gt[file_name],
                "gt_label_set": gt_multi[file_name],
            })

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        yield from self.items


class GazeHOITestSet:
    """One item per test instance (an annotated_file, e.g.
    "1159436__ann_00214.jpg"). The same raw photo can host several
    instances, one per person/object pair, each with its own visual
    prompt, label, and object ground-truth box."""

    def __init__(self, raw_images_dir: str, eval_csv: str, annotations_csv: str):
        self.raw_images_dir = raw_images_dir
        gt = load_ground_truth(eval_csv, annotations_csv)

        self.items = []
        for annotated_file, entry in sorted(gt.items()):
            raw_image_path = os.path.join(raw_images_dir, entry["raw_file_name"])
            if not os.path.isfile(raw_image_path):
                continue
            self.items.append({
                "file_name": annotated_file,
                "raw_image_path": raw_image_path,
                "head_bbox": entry["head_bbox"],
                "gt_label": norm_text(entry["object"]),
                "gt_label_set": [norm_text(entry["object"])],  # GazeHOI has one label per instance
                "retrieval_key": entry["raw_file_name"],
                "object_bbox": entry["object_bbox"],
            })

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        yield from self.items
