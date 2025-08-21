# FastAvatar: Towards Unified Fast High-Fidelity 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers

<div align="center">
<a href="https://tyrionwuyue.github.io/project_FastAvatar/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

[Yue Wu](https://tyrionwuyue.github.io/), [Yufan Wu](https://github.com/EavianWoo), [Wen Li](https://github.com/lililuya),  [Yuxi Lu](orcid.org/0000-0003-1205-3524), [Kairui Feng*](https://kelvinfkr.github.io/), [Xuanhong Chen*](https://github.com/neuralchen)

**Tongji University**; **Shanghai Innovation Institute**; **Shanghai Jiao Tong University**; **AKool Research**
</div>

<p align="center">
    <img src="./assets/images/teaser.png" width="100%">
</p>

## 🔥 Overview
FastAvatar is a **feedforward** 3D avatar framework capable of flexibly leveraging diverse daily recordings (e.g., a single image, multi-view observations, or monocular video) to reconstruct a high-quality 3D Gaussian Splatting (3DGS) model **within seconds**, using only a single **unified** model.

## 🎉 Core Highlights 
- **Unified Model with Flexible Multi-frame Aggregation for Ultra-high-fidelity Avatars**
- **Fast Feedforward Framework Delivering High-quality 3D Gaussian Splatting Models in Seconds**
- **Incremental 3D Avatar Reconstruction Leveraging Diverse Inputs (single-shot, monocular and multi-view)**

## 📹 Demo

### Self-reenacted
<div align="center">
    <img src="assets/images/self_001.gif">
</div>

### Cross-reenacted
<div align="center">
    <img src="assets/images/cross_001.gif">
</div>

### Multi-view & Incremental Reconstruction
<div align="center">
    <img src="assets/images/multiview_001.gif">
</div>

## 🚀 Get Started

### Enviroment Setup

```bash
git clone https://github.com/TyrionWuYue/FastAvatar.git
cd FastAvatar

conda create --name fastavatar python=3.10
conda activate fastavatar

bash ./scripts/install/install_cu124.sh
```

### HuggingFace Download
```bash
# Download Assets
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download TyrionWuY/FastAvatar-assets --local-dir ./tmp
tar -xf ./tmp/assets.tar 
tar -xf ./tmp/model_zoo.tar
rm -r ./tmp/
# Download Model Weihts
huggingface-cli download TyrionWuY/FastAvatar --local-dir ./model_zoo/fastavatar/
```

### Inference
```bash
bash scripts/infer/infer.sh ${CONFIG} ${MODEL_NAME} ${IMAGE_INPUT} ${MOTION_SEQS_DIR} ${INFERENCE_N_FRAMES} ${MODE}
```
`IMAGE_INPUT` can be either a .mp4 video file or a folder path containing arbitrary number of images. `INFERENCE_N_FRAMES` is used to control the number of frames input to the model. `MODE` has two options: 'Monocular' and 'MultiView'.


### Acknowledgement
This work is built on many amazing research works and open-source projects:
- [VGGT](https://github.com/facebookresearch/vggt)
- [LAM](https://github.com/aigc3d/LAM)
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
- [VHAP](https://github.com/ShenhanQian/VHAP)

Thanks for their excellent works and great contribution.