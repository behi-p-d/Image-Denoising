# Image-Denoising


A comparative study of CNN autoencoder architectures and loss functions for mixed-noise image denoising on the BSD500 dataset.

The project compares **Shallow, Deep, and Residual CNN Autoencoders** under mixed Gaussian, Salt-and-Pepper, and Speckle noise. The best-performing architecture is subsequently evaluated using both **MSE** and a **hybrid MSE–SSIM loss**.

## Overview

Image denoising aims to recover a clean image from a corrupted observation while preserving important structures and details.

This project investigates two main questions:

1. **Which CNN autoencoder architecture provides the best denoising performance?**
2. **Does a hybrid MSE–SSIM loss improve denoising performance compared with MSE?**

## Models

* **Shallow CNN Autoencoder**
* **Deep CNN Autoencoder**
* **Residual CNN Autoencoder**

## Dataset

The project uses the **BSD500** dataset for training, validation, and testing.

The dataset is organized into:


data/
└── images/
    ├── train/
    ├── val/
    └── test/


The dataset itself is **not included in this repository**. Please obtain BSD500 separately and place the images in the directory structure above.

Images are processed as 128 × 128 RGB images.

## Noise Model

To evaluate denoising under challenging conditions, images are corrupted using a combination of:

* **Gaussian noise**
* **Salt-and-Pepper noise**
* **Speckle noise**

The resulting mixed-noise images are used as inputs to the autoencoders, while the original clean images serve as reconstruction targets.

## Loss Functions

Two loss functions are investigated.

### Mean Squared Error (MSE)

MSE measures the pixel-wise difference between the reconstructed and clean images.

### Hybrid MSE–SSIM Loss

The hybrid loss combines pixel-level reconstruction accuracy with structural similarity:


Hybrid Loss = α × MSE + (1 − α) × (1 − SSIM)


where α = 0.8.

The hybrid loss is applied to the Residual CNN Autoencoder after selecting the best architecture using MSE.

## Evaluation Metrics

The models are evaluated using:

* **PSNR** — Peak Signal-to-Noise Ratio
* **SSIM** — Structural Similarity Index
* **Parameter Count**
* **Inference Time**

## Experimental Design

The experiments are organized into two stages.

### Stage 1 — Architecture Comparison

All three architectures are trained using MSE:


Shallow AE  ──┐
Deep AE     ──┼──► MSE ──► PSNR / SSIM
Residual AE ──┘


The results are compared to determine the best-performing architecture.

### Stage 2 — Loss Function Comparison

The best architecture, the Residual Autoencoder, is trained using the hybrid MSE–SSIM loss and compared with its MSE-trained counterpart:


Residual AE + MSE
        │
        ├──► PSNR
        ├──► SSIM
        └──► Inference Time

Residual AE + Hybrid MSE–SSIM
        │
        ├──► PSNR
        ├──► SSIM
        └──► Inference Time


This separates the effect of **architecture** from the effect of **loss function**.

## Visual Results

The project generates:

* Training loss curves
* Validation loss curves
* Denoising comparisons
* Reconstruction error maps

Generated results are stored in:


results/
├── histories/
├── plots/
└── tables/


A qualitative comparison contains:


Original | Noisy | Shallow AE | Deep AE | Residual AE | Residual AE (Hybrid)


## Project Structure


cnn-autoencoder-image-denoising/
│
├── data/
│   └── README.md
│
├── models/
│   ├── shallow_autoencoder.py
│   ├── deep_autoencoder.py
│   └── residual_autoencoder.py
│
├── utils/
│   ├── dataset.py
│   ├── noise.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── visualization.py
│
├── notebooks/
│   ├── 01_dataset.ipynb
│   ├── 02_testing.ipynb
│   ├── training.ipynb
│   └── results.ipynb
│
├── results/
│   ├── histories/
│   ├── plots/
│   └── tables/
│
├── config.py
├── experiment.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md


## Reproducibility

The project uses a fixed random seed:


SEED = 42


Training and evaluation settings are centralized in config.py.

The experiment pipeline saves:

* Best model checkpoints
* Training histories
* Evaluation results
* Generated plots
* Comparison tables

## Requirements

The main dependencies include:

* Python
* PyTorch
* Torchvision
* Torchmetrics
* NumPy
* SciPy
* Pillow
* scikit-image
* Matplotlib
* Pandas
* Jupyter

See requirements.txt for the complete dependency list and versions.

## License

This project is licensed under the **MIT License**. See the LICENSE file for details.

The BSD500 dataset is not distributed with this repository and remains subject to its own terms of use.

