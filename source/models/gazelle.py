"""GazeLLE gaze-point estimation: raw image + head bbox -> where they're looking."""
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class GazePrediction:
    x_px: float
    y_px: float
    inout: float
    heatmap: Any  # torch.Tensor, kept for the heatmap visual-prompt style


class GazeLLEEstimator:
    def __init__(self, torch_home: str, device: str = "cuda", repo: str = "fkryan/gazelle", model_name: str = "gazelle_dinov2_vitl14_inout"):
        os.environ.setdefault("TORCH_HOME", torch_home)
        import torch

        self.torch = torch
        self.device = device
        # torch.hub caches under torch_home. It loads from there if
        # already downloaded, or fetches it on first use.
        self.model, self.transform = torch.hub.load(repo, model_name, trust_repo=True)
        self.model.eval().requires_grad_(False).to(device)

    def predict(self, image, head_bbox_px) -> GazePrediction:
        W, H = image.size
        x1, y1, x2, y2 = head_bbox_px
        bbox_norm = (max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H))

        inp = {
            "images": self.transform(image).unsqueeze(0).to(self.device),
            "bboxes": [[bbox_norm]],
        }
        with self.torch.no_grad():
            out = self.model(inp)

        heatmap = out["heatmap"][0][0]
        inout = float(out["inout"][0][0]) if out.get("inout") is not None else 1.0

        Hh, Wh = heatmap.shape[-2], heatmap.shape[-1]
        idx = int(self.torch.argmax(heatmap.reshape(-1)).item())
        gx = (idx % Wh + 0.5) / Wh
        gy = (idx // Wh + 0.5) / Hh

        return GazePrediction(x_px=gx * W, y_px=gy * H, inout=inout, heatmap=heatmap)
