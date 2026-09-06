import csv
import os
import re

ANN_RE = re.compile(r"^(.+)__ann_(\d+)\.(\w+)$", re.IGNORECASE)


def load_ground_truth(eval_csv: str, semgaze_csv: str) -> dict:
    """Returns {annotated_file: {
        "raw_file_name": str,
        "head_bbox": (x1,y1,x2,y2), "object_bbox": (x1,y1,x2,y2),
        "object": str,
    }}
    """
    with open(semgaze_csv, newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    result = {}
    with open(eval_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            annotated_file = row["annotated_file"]
            ann_idx = int(row["ann_idx"])
            gt = gt_rows[ann_idx]

            m = ANN_RE.match(annotated_file)
            if not m:
                continue
            base, _, ext = m.groups()
            raw_file_name = f"{base}.{ext}"

            result[annotated_file] = {
                "raw_file_name": raw_file_name,
                "head_bbox": (
                    float(gt["h_xmin"]), float(gt["h_ymin"]),
                    float(gt["h_xmax"]), float(gt["h_ymax"]),
                ),
                "object_bbox": (
                    float(gt["o_xmin"]), float(gt["o_ymin"]),
                    float(gt["o_xmax"]), float(gt["o_ymax"]),
                ),
                "object": gt["object"],
            }
    return result
