<h1 align="center">From Gaze to Meaning: A Training-Free<br/>AI Agent for Unified Grounding and Explanation</h1>
<p align="center"><b>Shayan Nasiriboukani, Sara Atito, Mohammad Nezamipour, Muhammad Awais</b><br/>
Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, UK</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red.svg?logo=arXiv" alt="arXiv"></a>&nbsp;
  <a href="https://eccv.ecva.net/virtual/2026/poster/5517"><img src="https://img.shields.io/badge/ECCV-2026-blue.svg" alt="ECCV 2026"></a>&nbsp;
  <a href="https://shayan137.github.io/gta-project-page/"><img src="https://img.shields.io/badge/Project-Page-orange.svg" alt="Project Page"></a>
</p>

**Gaze Target Agent (GTA)** is, to our knowledge, the first fully **training-free agent for gaze-guided visual reasoning**. It uses gaze to guide a vision-language model toward the attended region, identifies the gaze target, and optionally grounds it with a bounding box.
GTA combines **gaze-guided visual prompting**, **uncertainty-aware retrieval**, and **object grounding**, using only frozen pretrained models and no fine-tuning.

## Key Points

* **Training-free:** No fine-tuning or parameter updates.
* **Gaze-guided reasoning:** Gaze provides an explicit visual cue to the VLM.
* **Uncertainty-aware retrieval:** Similar examples are retrieved only when the prediction is uncertain.
* **Unified prediction and grounding:** GTA identifies the attended object and can localize it in the image.

## Method

<p align="center">
  <img src="figs/architecture.png" width="100%" alt="Gaze Target Agent architecture">
</p>


GTA follows a simple training-free pipeline. It first estimates the gaze target and converts it into a visual prompt. The VLM then predicts the attended object and estimates its confidence. If the prediction is uncertain, similar examples are retrieved from memory and used to refine the answer. For GazeHOI, GTA can additionally ground the predicted object with a bounding box.



## Installation

```bash
git clone https://github.com/shayan137/Gaze-Target-Agent.git
cd Gaze-Target-Agent
pip install -r requirements.txt
```

The required models, including **Qwen3-VL, GazeLLE, CLIP, and the grounding detector**, are downloaded automatically on first use into the directory specified by `model_cache_dir`.

## 📦 Data Preparation

Download the two benchmarks:

- **[GazeFollow](https://huggingface.co/datasets/vikhyatk/gazefollow)**: images, head bounding-box annotations, and gaze-target labels.
- **[GazeHOI](https://github.com/idiap/semgaze)**: images, head/object annotations, and object labels.

Then update the dataset paths in `configs/gazefollow.yaml` or `configs/gazehoi.yaml`.


## 🔨 Usage

### GazeFollow

```bash
python scripts/run_agent.py --config configs/gazefollow.yaml
```

This runs the full pipeline:

```text
Gaze Estimation
→ Visual Prompting
→ Qwen3-VL Prediction
→ Uncertainty Check
→ Retrieval Rescue, if needed
```

On the first run, GTA also builds and saves the training CLIP embeddings used for retrieval. Later runs load them directly.

### GazeHOI

```bash
python scripts/run_gazehoi_agent.py --config configs/gazehoi.yaml
```

GazeHOI follows the same pipeline and adds **object grounding** to localize the predicted target with a bounding box.

To manually rebuild the retrieval embeddings, for example after changing the CLIP model:

```bash
python scripts/build_cls_embeddings.py --config configs/gazefollow.yaml
```

This step is optional because the main scripts build them automatically when needed.


**GazeFollow:**

```bash
python scripts/run_agent.py --config configs/gazefollow.yaml \
  --raw-images-dir /path/to/GazeFollow/train_test_images \
  --annotations-txt /path/to/GazeFollow/train_test_images/test_annotations_release.txt \
  --gt-csv /path/to/GazeFollow/gaze-labels-test.csv \
  --vocab-path /path/to/GazeFollow/vocab.json \
  --train-labels-csv /path/to/GazeFollow/gaze-labels-train.csv
```

**GazeHOI:**

```bash
python scripts/run_gazehoi_agent.py --config configs/gazehoi.yaml \
  --raw-images-dir /path/to/GazeHOI/images \
  --eval-csv /path/to/GazeHOI/test-annotations.csv \
  --annotations-csv /path/to/GazeHOI/annotations.csv \
  --vocab-path /path/to/GazeHOI/vocab.json \
  --train-annotations-csv /path/to/GazeHOI/train-annotations.csv
```

Any omitted argument falls back to the value in the YAML file.

Useful optional arguments include:

```text
--limit N
```

Runs only the first `N` samples for quick testing.


## 🗂️ Code Structure
```
source/
  agent.py                 # GazeTargetAgent.run(): predict -> check -> retrieve
  model.py                 # Qwen3-VL wrapper
  gazelle.py                # gaze estimation
  visual_arrow.py           # gaze-guided visual prompting (arrow/dot/heatmap)
  retrieval.py              # CLIP CLS retrieval (RAG rescue)
  grounding.py              # detector + gaze-guided selection + scoring (GazeHOI)
  dataset.py, config.py, ...
scripts/
  build_cls_embeddings.py  # one-time per dataset: CLIP embeddings for training images
  run_agent.py              # GazeFollow entrypoint
  run_gazehoi_agent.py       # GazeHOI entrypoint (adds grounding)
configs/
  gazefollow.yaml, gazehoi.yaml  # one config per dataset
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
