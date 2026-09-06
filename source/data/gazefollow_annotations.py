import csv
import os


def load_head_bboxes(annotations_txt: str) -> dict:
    """Returns {file_name: {"head_bbox": (x1,y1,x2,y2), "rel_path": image_path}}."""
    result = {}
    with open(annotations_txt, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            rel_path = row[0]
            file_name = os.path.basename(rel_path)
            if file_name in result:
                continue
            x1, y1, x2, y2 = (float(v) for v in row[10:14])
            result[file_name] = {"head_bbox": (x1, y1, x2, y2), "rel_path": rel_path}
    return result
