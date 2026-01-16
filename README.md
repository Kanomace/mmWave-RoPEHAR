# RoPEHAR: A Real-Time Rotary Position Encoding Informer for mmWave-Based Human Activity Recognition in Substations

This repository provides the **official implementation, dataset, and GUI demo** for the paper:

> **RoPEHAR: A Real-Time Rotary Position Encoding Informer for mmWave-Based Human Activity Recognition in Substations**  
> *IEEE Internet of Things Journal (under revision)*

RoPEHAR is a **millimeter-wave radar based human activity recognition (HAR) system** designed for **real-world power substation environments**, addressing challenges such as electromagnetic interference, human–equipment coupling, and real-time constraints.

---

## 📌 Highlights

- 📡 **mmWave radar based** (TI IWR1843BOOST)
- 🧠 **Roformer**: Informer + Rotary Position Encoding (RoPE)
- 🧹 **Hybrid SNR–DBSCAN denoising** for EMI-robust point cloud extraction
- ⚡ **Real-time inference** for safety-critical substation operations
- 🗂 **Public dataset + reproducible pipeline**
- 🖥 **Interactive GUI** for visualization and inference

---

## 📐 System Overview

<p align="center">
  <img src="fig/framework.png" width="85%">
</p>

**RoPEHAR pipeline:**

1. Raw FMCW radar signals → TLV packets  
2. Hybrid denoising (SNR filtering + DBSCAN clustering)  
3. Environmental voxelization and dimensionality reduction  
4. Roformer-based spatiotemporal classification  
5. Real-time action recognition (10 classes)

---

## 🧠 Roformer Architecture

<p align="center">
  <img src="fig/roformer.png" width="80%">
</p>

**Key features:**

- ProbSparse Attention (Informer) for long-sequence efficiency  
- Rotary Position Encoding (RoPE) for relative spatiotemporal modeling  
- Dual-plane voxel projection (XOZ / YOZ)  
- Cross-view feature fusion  

---

## 📁 Repository Structure
mmWave-RoPEHAR/
│
├── binData/ # Dataset (raw & processed)
│ ├── traindata/
│ │ ├── 0static
│ │ ├── 1squat
│ │ ├── 2stand
│ │ ├── 3tumble
│ │ ├── 4open
│ │ ├── 5Switch
│ │ ├── 6close
│ │ ├── 7circle
│ │ ├── 8Rcircle
│ │ └── 9sign
│ └── processed_data/
│
├── rope_informer/ # RoPEHAR core model
│ ├── models/
│ ├── exp/
│ ├── scripts/
│ ├── utils/
│ └── results/
│
├── model_checkpoint/ # Pretrained weights
├── results/ # Experimental results & logs
├── inference_outputs/ # Inference visualizations
│
├── gui/ # GUI-based visualization & demo
│ ├── assets/
│ ├── images/
│ ├── docs/
│ └── history/
│
├── fig/ # Figures used in paper / README
└── README.md

---

## 📊 Dataset Description

- **Sensor**: TI IWR1843BOOST mmWave radar  
- **Environment**: Real-world indoor power substation  
- **Participants**: 4 electrical workers (2 male, 2 female)  
- **Actions**: 10 typical electrical maintenance operations  
- **Total samples**: 24,000  
- **Frame rate**: 10 FPS  
- **Annotation**: Frame-level action labels  

### Action Classes

| ID | Action Name |
|----|-------------|
| 0  | Static posture |
| 1  | Squatting |
| 2  | Standing |
| 3  | Falling |
| 4  | Opening cabinet door |
| 5  | Operating switch |
| 6  | Closing cabinet door |
| 7  | Trolley swinging |
| 8  | Rotational operation |
| 9  | Hanging safety sign |

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
conda create -n ropehar python=3.9
conda activate ropehar
pip install -r requirements.txt
