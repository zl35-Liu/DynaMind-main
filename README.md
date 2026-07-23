# DynaMind: Reconstructing Dynamic Visual Scenes from EEG by Aligning Temporal Dynamics and Multimodal Semantics to Guided Diffusion

<div align="center">

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-4b44ce.svg)](https://openaccess.thecvf.com/content/CVPR2026F/html/Liu_DynaMind_Reconstructing_Dynamic_Visual_Scenes_from_EEG_by_Aligning_Temporal_CVPRF_2026_paper.html)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://www.jiangtongli.me/uploads/papers/2026-CVPR-DynaMind.pdf)
[![Code GitHub](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github)](https://github.com/zl35-Liu/DynaMind-main)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f.svg)](https://zl35-liu.github.io/DynaMind/)

*A neuroscience-inspired dual-guidance framework for reconstructing semantically accurate and temporally coherent dynamic visual scenes from EEG.*

[Overview](#overview) • [Results](#results-at-a-glance) • [Repository Structure](#repository-structure) • [Installation](#installation) • [Citation](#citation) • [Acknowledgments](#acknowledgments)

</div>

---

<div align="center">
  <img src="https://zl35-liu.github.io/DynaMind/static/images/paper/compare1.png" alt="Qualitative comparison of DynaMind and EEG2Video on SEED-DV" width="95%">
  <p><strong>Qualitative comparison of DynaMind and EEG2Video reconstructions against ground-truth videos on SEED-DV.</strong></p>
</div>

DynaMind reconstructs dynamic visual scenes from electroencephalography (EEG) by jointly modeling regional visual semantics and neural temporal dynamics. The framework follows the neuroscience-inspired dual-stream theory and uses semantic and temporal signals to guide video diffusion.

## Overview

DynaMind contains three complementary modules:

| Module | Purpose | Implementation |
| --- | --- | --- |
| Regional-aware Semantic Mapper (RSM) | Processes frontal, temporal, parietal, and occipital EEG regions, models dual-stream interactions with channel-wise gating, and maps multimodal EEG features into a semantic diffusion prior. | [`moudules/RSM`](moudules/RSM) |
| Temporal-aware Dynamic Aligner (TDA) | Converts windowed EEG signals into a temporal blueprint and aligns it with video latents through frame-wise content and structural consistency objectives. | [`moudules/TDA`](moudules/TDA) |
| Dual-Guidance Video Reconstructor (DGVR) | Uses the RSM semantic prior and TDA temporal blueprint to condition latent video diffusion for semantically accurate and temporally coherent reconstruction. | [`moudules/DGVR`](moudules/DGVR) |

The modules are trained independently and combined during reconstruction. The paper evaluates a Stable Diffusion V1.4-based configuration on SEED-DV and a CogVideoX-based configuration on CineBrain.

<div align="center">
  <img src="https://zl35-liu.github.io/DynaMind/static/images/paper/Framework2.png" alt="Overview of the DynaMind framework" width="95%">
</div>

## Results at a Glance

The paper evaluates DynaMind on SEED-DV and CineBrain using semantic, perceptual, and temporal metrics.

| Dataset | Metric | DynaMind | Reference method |
| --- | ---: | ---: | ---: |
| SEED-DV, 40 classes | Video 40-way accuracy (higher is better) | **0.284** | 0.159 (EEG2Video) |
| SEED-DV, 40 classes | FVMD (lower is better) | **1637.55** | 2038.27 (EEG2Video) |
| CineBrain | Video 50-way accuracy (higher is better) | **0.317** | 0.304 (CineSync-EEG) |
| CineBrain | FVD (lower is better) | **51.50** | 53.75 (CineSync-EEG) |

See the [paper](https://www.jiangtongli.me/uploads/papers/2026-CVPR-DynaMind.pdf) and [project page](https://zl35-liu.github.io/DynaMind/) for complete quantitative results, ablations, and qualitative comparisons.

## Repository Structure

```text
DynaMind-main/
|-- configs/                 Experiment configurations for RSM, TDA, and DGVR
|-- moudules/
|   |-- RSM/                 Regional-aware semantic mapping and diffusion prior
|   |-- TDA/                 Temporal blueprint learning and alignment
|   `-- DGVR/                Dual-guidance video diffusion
|-- eval/                    Semantic, perceptual, temporal, and visualization tools
|-- utils/                   Shared data, training, and visualization utilities
|-- inference.py             SEED-DV reconstruction entry point
|-- inference_cine.py        CineBrain reconstruction entry point
`-- requirements.txt         Python dependencies
```

## Installation

A CUDA-capable environment is recommended. Install a PyTorch build compatible with the local CUDA runtime before installing the remaining dependencies.

```bash
git clone https://github.com/zl35-Liu/DynaMind-main.git
cd DynaMind-main

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Citation

If this project is useful for your research, please cite:

```bibtex
@inproceedings{liu2026dynamind,
  title     = {DynaMind: Reconstructing Dynamic Visual Scenes from EEG by Aligning Temporal Dynamics and Multimodal Semantics to Guided Diffusion},
  author    = {Liu, Junxiang and Lin, Junming and Zhou, Jie and Xiong, Wei and Ji, Hongfei and Zhuang, Jie and Li, Jie and Li, Jiangtong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

## Acknowledgments

This project builds on research and open-source work in EEG decoding, CLIP-based multimodal alignment, latent diffusion, Tune-A-Video, Stable Diffusion, and CogVideoX. Please also cite the corresponding datasets and foundational methods when using this repository.
