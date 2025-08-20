# FastAvatar: Towards Unified Fast High-Fidelity 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers

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

### Acknowledgement
This work is built on many amazing research works and open-source projects:
- [VGGT](https://github.com/facebookresearch/vggt)
- [LAM](https://github.com/aigc3d/LAM)
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
- [VHAP](https://github.com/ShenhanQian/VHAP)

Thanks for their excellent works and great contribution.