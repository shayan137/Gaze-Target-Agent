<h2 align="center">From Gaze to Meaning: A Training-Free<br/>AI Agent for Unified Grounding and Explanation</h2>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
  <!-- TODO: replace XXXX.XXXXX above with the real arXiv id once assigned -->
  <a href="https://shayan137.github.io/gta-project-page/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/ECCV-2026-4b44ce.svg" alt="ECCV 2026">
</p>

<p align="center"><b>Shayan Nasiriboukani, Sara Atito, Mohammad Nezamipour, Muhammad Awais</b><br/>
Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, UK</p>

---

**Gaze Target Agent (GTA)** is the first **training-free** agent for
gaze-guided reasoning: gaze target prediction, attention localization, and object
identification, all from a single pretrained vision-language model. GTA augments a frozen
VLM with **gaze-guided visual prompting**, rescues low-confidence predictions with a
**memory-based (RAG) retrieval** step, and **grounds** the predicted label to a
bounding box. GTA reaches state-of-the-art results on the GazeFollow and GazeHOI benchmarks.

<p align="center">
  <img src="figs/architecture.png" width="100%" alt="Gaze Target Agent architecture">
</p>

## 🛠️ Installation
```bash
cd GazeTargetAgent-main
pip install -r requirements.txt
```
Every model (Qwen3-VL, GazeLLE, CLIP, the grounding detector) auto-downloads on first use
into one shared, repo-local folder, `model_cache_dir`.

## 📦 Data Preparation
Download the two benchmarks and point each dataset's `configs/*.yaml` at where you put
them (`raw_images_dir`, the annotation files, `vocab_path`):
- **[GazeFollow](https://huggingface.co/datasets/vikhyatk/gazefollow)** -- raw images +
  gaze annotations.
- **[GazeHOI](https://github.com/idiap/semgaze)** -- raw images, head/object-bbox
  annotations, **and the object label files for both datasets**.

## 🔨 Usage
```bash
python scripts/build_cls_embeddings.py --config configs/gazefollow.yaml
python scripts/run_agent.py --config configs/gazefollow.yaml
```

For GazeHOI, the same pass adds a grounding step, localizing the prediction to a
bounding box:
```bash
python scripts/build_cls_embeddings.py --config configs/gazehoi.yaml
python scripts/run_gazehoi_agent.py --config configs/gazehoi.yaml
```

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
  gazefollow.yaml, gazehoi.yaml
```

## 📑 Citation
```bibtex
@inproceedings{nasiriboukani2026gaze,
  title     = {From Gaze to Meaning: A Training-Free AI Agent for Unified Grounding and Explanation},
  author    = {Nasiriboukani, Shayan and Atito, Sara and Nezamipour, Mohammad and Awais, Muhammad},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
