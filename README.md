# Enhanced Spatiotemporal Transformer for Facial Action Unit Detection Using Multi-Scale Pyramid Attention and Cross-Channel Feature Fusion
[![DOI](https://zenodo.org/badge/1179545482.svg)](https://doi.org/10.5281/zenodo.19124244)

This repository contains the official PyTorch implementation of the paper **"Enhanced Spatiotemporal Transformer for Facial Action Unit Detection Using Multi-Scale Pyramid Attention and Cross-Channel Feature Fusion"**, submitted to ***The Visual Computer***.

Our proposed model introduces a lightweight, plug-and-play feature extraction enhancement strategy that effectively captures subtle and multi-scale facial muscle deformations in real-world scenarios, setting a new benchmark for AU detection.

## Requirements
* torch 1.6.0
* torchaudio 0.6.0
* tqdm
* Numpy
* OpenCV 4.2.0
* lmdb
* einops

## Dataset Preparation
Our model is trained and evaluated on the Aff-Wild2 dataset.
* **Official Dataset:** Please download the official raw dataset from the [Aff-Wild2 Official Website](https://sites.google.com/view/dimitrioskollias/databases/aff-wild2).
* **Submission Files & Pre-processed Data:** To easily reproduce our results, download the required submission files from Baidu Netdisk:
  * **Link:** https://pan.baidu.com/s/19oIi6qoGRAPifTb0hS3aDA?pwd=nfke
  * **Extraction Code:** `nfke`

## Usage
Running the model is very straightforward. Please follow these steps:
1. Download the dataset and place it into the `submission` folder.
2. Change the dataset path in the configuration to your actual local path.
3. Select the model name in the `opt` settings.
4. Run the training script:
   ```bash
   python train.py
