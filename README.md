<h3 align="center">From Gaze to Meaning: A Training-Free<br/>AI Agent for Unified Grounding and Explanation</h3>
<p align="center"><b>Shayan Nasiriboukani, Sara Atito, Mohammad Nezamipour, Muhammad Awais</b><br/>
Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, UK</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red.svg?logo=arXiv" alt="arXiv">
  </a>
  <a href="https://eccv.ecva.net/virtual/2026/poster/5517">
    <img src="https://img.shields.io/badge/ECCV-2026-blue.svg" alt="ECCV 2026">
  </a>
  <a href="https://shayan137.github.io/gta-project-page/">
    <img src="https://img.shields.io/badge/Project-Page-orange.svg" alt="Project Page">
  </a>
</p>

**Gaze Target Agent (GTA)** is, to our knowledge, the first fully **training-free agent for gaze-guided visual reasoning**. It uses gaze to guide a vision-language model toward the attended region, identifies the gaze target, and optionally grounds it with a bounding box.

GTA combines **gaze-guided visual prompting**, **uncertainty-aware retrieval**, and **object grounding**, using only frozen pretrained models and no fine-tuning.

## Key Points

* **Training-free:** No fine-tuning or parameter updates.
* **Gaze-guided reasoning:** Gaze provides an explicit visual cue to the VLM.
* **Uncertainty-aware retrieval:** Similar examples are retrieved only when the prediction is uncertain.
* **Unified prediction and grounding:** GTA identifies the attended object and can localize it in the image.

<p align="center">
  <img src="figs/architecture.png" width="100%" alt="Gaze Target Agent architecture">
</p>

## Method

GTA follows a simple training-free pipeline. It first estimates the gaze target and converts it into a visual prompt. The VLM then predicts the attended object and estimates its confidence. If the prediction is uncertain, similar examples are retrieved from memory and used to refine the answer. For GazeHOI, GTA can additionally ground the predicted object with a bounding box.

```text
Image + Head
     ↓
Gaze Estimation
     ↓
Gaze-Guided Visual Prompt
     ↓
VLM Prediction + Uncertainty
     ↓
High uncertainty?
   ↙       ↘
 Yes       No
  ↓         ↓
Retrieval   │
+ Re-predict│
   ↘       ↙
 Final Object Prediction
         ↓
 Optional Object Grounding
```

## Installation

```bash
git clone https://github.com/shayan137/Gaze-Target-Agent.git
cd Gaze-Target-Agent
pip install -r requirements.txt
```

The required models, including **Qwen3-VL, GazeLLE, CLIP, and the grounding detector**, are downloaded automatically on first use into the directory specified by `model_cache_dir`.

Models can be changed directly in the configuration file. For example:

```yaml
qwen_model: Qwen3-VL-4B-Instruct
```

## Data Preparation

Download the benchmarks and update their paths in `configs/*.yaml`.

* **[GazeFollow](https://huggingface.co/datasets/vikhyatk/gazefollow):** images, head annotations, target labels, and vocabulary.
* **[GazeHOI](https://github.com/idiap/semgaze):** images, head/object annotations, target labels, and vocabulary.

Main configuration fields:

```text
annotations_txt / annotations_csv   Head annotations
gt_csv / eval_csv                   Ground-truth target labels
vocab_path                          Target-object vocabulary
```

## Usage

### GazeFollow

```bash
python scripts/run_agent.py --config configs/gazefollow.yaml
```

This runs:

```text
Gaze Estimation
→ Visual Prompting
→ VLM Prediction
→ Uncertainty Estimation
→ Retrieval Rescue, if needed
```

The retrieval embeddings are automatically created on the first run and reused afterward. This only extracts features from a frozen CLIP model; **no training is performed**.

### GazeHOI

```bash
python scripts/run_gazehoi_agent.py --config configs/gazehoi.yaml
```

This runs the same pipeline with an additional **object-grounding** stage.

To manually rebuild the retrieval embeddings, for example after changing the CLIP model:

```bash
python scripts/build_cls_embeddings.py --config configs/gazefollow.yaml
```

## Code Structure

```text
source/
  agent.py                   # prediction and retrieval rescue
  config.py                  # YAML configuration
  prompts.py                 # VLM prompts
  threshold.py               # uncertainty threshold
  uncertainty.py             # confidence estimation
  visual_arrow.py            # gaze-guided visual prompts

  data/
    dataset.py               # dataset loaders
    gazefollow_annotations.py
    gazehoi_annotations.py
    vocab.py
    io_utils.py
    text_utils.py

  models/
    model.py                 # Qwen3-VL
    gazelle.py               # gaze estimation
    retrieval.py             # CLIP retrieval
    grounding.py             # object grounding

scripts/
  build_cls_embeddings.py
  run_agent.py
  run_gazehoi_agent.py
  reground.py

configs/
  gazefollow.yaml
  gazehoi.yaml
```

## Citation

```bibtex
@inproceedings{nasiriboukani2026gaze,
  title     = {From Gaze to Meaning: A Training-Free AI Agent for Unified Grounding and Explanation},
  author    = {Nasiriboukani, Shayan and Atito, Sara and Nezamipour, Mohammad and Awais, Muhammad},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
