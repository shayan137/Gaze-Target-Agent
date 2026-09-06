<h2 align="center">From Gaze to Meaning: A Training-Free<br/>AI Agent for Unified Grounding and Explanation</h2>

<p align="center">
  <b>Shayan Nasiriboukani, Sara Atito, Mohammad Nezamipour, Muhammad Awais</b><br/>
  Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, UK
</p>

<p align="center">
  <a href="https://eccv.ecva.net/virtual/2026/poster/5517">
    <img src="https://img.shields.io/badge/ECCV-2026-4b44ce.svg" alt="ECCV 2026">
  </a>
  <a href="https://shayan137.github.io/gta-project-page/">
    <img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page">
  </a>
</p>

---

**Gaze Target Agent (GTA)** is, to our knowledge, the first fully **training-free agent for gaze-guided visual reasoning**. Given an image and a person's gaze, GTA goes beyond predicting *where* the person is looking: it identifies *what* they are looking at and can ground that object in the image.

GTA combines frozen pretrained models with **gaze-guided visual prompting**, **uncertainty-aware retrieval**, and **object grounding**. The predicted gaze is converted into a visual cue that guides a vision-language model (VLM) toward the attended region. When the VLM is uncertain, GTA retrieves similar gaze-conditioned examples from a memory bank and uses them as in-context examples to refine the prediction.

The entire pipeline works **without fine-tuning or parameter updates**. GTA achieves state-of-the-art results on **GazeFollow** and **GazeHOI**, while also supporting flexible open-vocabulary predictions.

## ✨ Highlights

* 🎯 **Fully training-free.** Uses frozen pretrained models with no fine-tuning.
* 👁️ **Gaze guides the VLM.** Gaze is turned into an explicit visual prompt.
* 🧠 **Retrieval only when needed.** Uncertain predictions are refined using similar examples.
* 📍 **From gaze to meaning.** GTA identifies and grounds the attended object.

<p align="center">
  <img src="figs/architecture.png" width="100%" alt="Gaze Target Agent architecture">
</p>

## 🔍 How GTA Works

GTA processes each gaze target in four main stages:

1. **Estimate gaze.**
   The person's head location is provided to GazeLLE, which predicts where the person is looking.

2. **Guide the VLM.**
   The predicted gaze is drawn on the image as a visual prompt. Qwen3-VL then predicts the attended object and produces its top candidate answers.

3. **Refine uncertain predictions.**
   GTA measures the confidence of the VLM prediction. If confidence is low, it retrieves similar gaze-conditioned examples from a memory bank and uses them as in-context examples for a second prediction.

4. **Ground the target.**
   For GazeHOI, the predicted object is matched to a candidate region using the gaze signal, producing a bounding box for the attended object.

This gives GTA a simple inference pipeline:

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

## 🛠️ Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

The required models, including **Qwen3-VL, GazeLLE, CLIP, and the grounding detector**, are downloaded automatically on first use.

All models share the cache directory defined by `model_cache_dir`, so no manual model download is required.

Models can be changed directly from the dataset configuration file. For example:

```yaml
qwen_model: Qwen3-VL-4B-Instruct
```

## 📦 Data Preparation

Download the benchmarks and update the corresponding paths in `configs/*.yaml`.

Each dataset requires the raw images, head annotations, target-object labels, and the target vocabulary used for evaluation.

* **[GazeFollow](https://huggingface.co/datasets/vikhyatk/gazefollow)**
  Raw images, head bounding-box annotations, target-object labels, and vocabulary.

* **[GazeHOI](https://github.com/idiap/semgaze)**
  Raw images, head/object bounding-box annotations, target-object labels, and vocabulary.

The main configuration fields are:

```text
annotations_txt / annotations_csv   Head annotations
gt_csv / eval_csv                   Ground-truth target labels
vocab_path                          Valid target-object vocabulary
```

## 🚀 Usage

### GazeFollow

```bash
python scripts/run_agent.py --config configs/gazefollow.yaml
```

This runs the full GTA pipeline for each test sample:

```text
Gaze estimation
→ Visual prompting
→ Qwen3-VL prediction
→ Uncertainty estimation
→ Retrieval rescue when needed
```

If the retrieval memory does not already exist, GTA automatically computes the **CLIP image embeddings** from the training set and saves them for later use.

> **Note:** building these embeddings does **not** train or fine-tune any model. It only extracts and stores features from a frozen CLIP encoder for retrieval.

Future runs reuse the saved embeddings directly.

### GazeHOI

```bash
python scripts/run_gazehoi_agent.py --config configs/gazehoi.yaml
```

GazeHOI uses the same reasoning pipeline and additionally performs **object grounding** to localize the predicted gaze target with a bounding box.

### Rebuild the Retrieval Memory

To manually rebuild the training-image embeddings, for example after changing `clip_model`, run:

```bash
python scripts/build_cls_embeddings.py --config configs/gazefollow.yaml
```

This step is optional. The main agent automatically builds the embeddings when they are missing.

## 🗂️ Code Structure

```text
source/
  agent.py                   # GTA prediction and retrieval-rescue logic
  config.py                  # YAML configuration
  prompts.py                 # VLM prompt templates
  threshold.py               # uncertainty threshold policy
  uncertainty.py             # prediction confidence estimation
  visual_arrow.py            # gaze-guided visual prompts

  data/
    dataset.py               # dataset loaders
    gazefollow_annotations.py
    gazehoi_annotations.py
    vocab.py                 # target-object vocabulary
    io_utils.py
    text_utils.py

  models/
    model.py                 # Qwen3-VL wrapper
    gazelle.py               # gaze estimation
    retrieval.py             # CLIP-based memory retrieval
    grounding.py             # gaze-guided object grounding

scripts/
  build_cls_embeddings.py    # build retrieval memory
  run_agent.py               # GazeFollow pipeline
  run_gazehoi_agent.py       # GazeHOI pipeline + grounding
  reground.py                # re-run grounding without VLM inference

configs/
  gazefollow.yaml
  gazehoi.yaml
```

## 📑 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{nasiriboukani2026gaze,
  title     = {From Gaze to Meaning: A Training-Free AI Agent for Unified Grounding and Explanation},
  author    = {Nasiriboukani, Shayan and Atito, Sara and Nezamipour, Mohammad and Awais, Muhammad},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
