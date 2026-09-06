"""YAML-based configuration for the gaze-target agent.

Each dataset uses a single configuration file that defines the complete
pipeline, including dataset-specific paths for training and test data,
model identifiers for CLIP, GazeLLE, Qwen, and the detector, a shared
model cache directory, visual-prompt rendering options, prediction and
rescue settings, and, for GazeHOI, grounding configuration.
"""

import dataclasses
import yaml


def load_yaml(cls, path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return cls(**data)


@dataclasses.dataclass
class AgentConfig:
    # Data (test/inference side)
    raw_images_dir: str
    annotations_txt: str  # head-bbox annotations
    gt_csv: str
    vocab_path: str

    # Data (train side, for scripts/build_cls_embeddings.py)
    train_labels_csv: str

    # CLS retrieval (see scripts/build_cls_embeddings.py)
    train_embeds: str
    train_meta: str

    output_dir: str

    dataset: str = "gazefollow"

    # One shared folder every auto-downloaded model caches into.
    model_cache_dir: str = ""


    qwen_model: str = "Qwen3-VL-2B-Instruct"
    gazelle_repo: str = "fkryan/gazelle"
    gazelle_model: str = "gazelle_dinov2_vitl14_inout"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"

    # Visual-prompt drawing style
    style: str = "arrow"  # arrow | dot | heatmap
    color: str = "blue"  # red | yellow | green | blue | [R,G,B]
    arrow_width: int = 4
    dot_radius: int = 6
    heatmap_alpha: float = 0.5


    query_images_dir: str = ""
    save_visual_prompts: bool = True

    top_n: int = 3
    rag_top_k: int = 6
    temperature: float = 0.0
    max_new_tokens_stage1: int = 32
    max_new_tokens_stage2: int = 32

    # A number (e.g. 0.25), or "dynamic". See source/threshold.py.
    uncertainty_threshold: object = 0.25
    dynamic_batch_size: int = 10
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        return load_yaml(cls, path)


@dataclasses.dataclass
class GazeHOIConfig:
    # Data (test/inference side)
    raw_images_dir: str             # holds the train images
    eval_csv: str                   # GazeHOI_test-annotations.csv (annotated_file, ann_idx, gt_object)
    annotations_csv: str            # semgaze-data test-annotations.csv (head bbox, object GT bbox)
    vocab_path: str

    # Data (train side, for scripts/build_cls_embeddings.py)
    train_annotations_csv: str

    # CLS retrieval (see scripts/build_cls_embeddings.py)
    train_embeds: str
    train_meta: str
    label_column: str = "gt_object"

    dataset: str = "gazehoi"
    output_dir: str = ""

    model_cache_dir: str = ""

    qwen_model: str = "Qwen3-VL-2B-Instruct"
    gazelle_repo: str = "fkryan/gazelle"
    gazelle_model: str = "gazelle_dinov2_vitl14_inout"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"

    style: str = "arrow"  # arrow | dot | heatmap
    color: str = "blue"  # red | yellow | green | blue | [R,G,B]
    arrow_width: int = 4
    dot_radius: int = 6
    heatmap_alpha: float = 0.5

    query_images_dir: str = ""
    save_visual_prompts: bool = True
    save_heatmaps: bool = False  # needed by scripts/reground.py to re-score without re-running the VLM

    top_n: int = 3
    rag_top_k: int = 6
    temperature: float = 0.0
    max_new_tokens_stage1: int = 32
    max_new_tokens_stage2: int = 32

    # A number (e.g. 0.25), or "dynamic". See source/threshold.py.
    uncertainty_threshold: object = 0.25
    dynamic_batch_size: int = 10
    seed: int = 42

    # Grounding (GazeHOI only)
    grounding_enabled: bool = True
    detector: str = "rfdetr"  # rfdetr | yolo26 | sam3
    rfdetr_variant: str = "2xlarge"  # 2xlarge | seg2xlarge
    yolo26_weights: str = "yolo26n.pt"
    sam3_checkpoint: str = "facebook/sam3"
    iou_threshold: float = 0.5
    detector_confidence_threshold: float = 0.5
    grounding_mode: str = "sqrt_norm_heatmap"  # gaze_point | heatmap | sqrt_norm_heatmap | all

    @classmethod
    def from_yaml(cls, path: str) -> "GazeHOIConfig":
        return load_yaml(cls, path)
