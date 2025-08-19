# FastAvatar: Towards Unified Fast High-Fidelity 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers

<p align="center">
    <img src="./assets/images/teaser.png" width="100%">
</p>

## 🔥 Overview
FastAvatar is a **feedforward** 3D avatar framework capable of flexibly leveraging diverse daily recordings (e.g., a single image, multi-view observations, or monocular video) to reconstruct a high-quality 3D Gaussian Splatting (3DGS) model **within seconds**, using only a single **unified** model.

## 📹 Demo

### Self-reenacted

### Cross-reenacted
<div align="center">
    <img src="assets/images/cross_001.gif">
</div>

### Multi-view

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