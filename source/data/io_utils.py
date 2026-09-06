"""CSV result read/write helpers, including basic resume support."""
import csv
import dataclasses
import os


FIELDS = [
    "file_name",
    "gt_label",
    "gt_label_set",
    "stage1_preds",
    "stage1_pred_prob",
    "stage1_uncertainty",
    "stage1_entropy",
    "stage1_top1_correct",
    "stage1_topn_correct",
    "stage1_multi_correct",
    "rag_applied",
    "rag_candidates",
    "rag_preds",
    "rag_top1_correct",
    "rag_topn_correct",
    "rag_multi_correct",
    "final_pred",
    "final_correct",
    "final_topn_correct",
    "final_multi_correct",
]


def result_to_row(result) -> dict:
    row = dataclasses.asdict(result)
    row["gt_label_set"] = "|".join(row["gt_label_set"])
    row["stage1_preds"] = "|".join(row["stage1_preds"])
    row["rag_candidates"] = "|".join(row["rag_candidates"]) if row["rag_candidates"] else ""
    row["rag_preds"] = "|".join(row["rag_preds"]) if row["rag_preds"] else ""
    return row


def write_results_csv(path: str, results: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_row(result))
