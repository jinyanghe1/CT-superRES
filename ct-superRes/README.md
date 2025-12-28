# CT Super-Resolution Framework

A simple PyTorch-based framework for CT image Super-Resolution using a ResNet architecture (SRResNet).

## Features

1. **High-Quality Dataset Module**: Automatically processes high-resolution CT images to create Low-Res/High-Res pairs for training. Supports **DICOM (.dcm)**, PNG, JPG, and TIF formats.
2. **Training Policy**: Standard SRResNet training with MSE loss. Customizable hyperparameters.
3. **Experiment & Evaluation**: Calculates PSNR, SSIM, MSE, and Inference Time. Generates visual comparisons and metric distribution graphs.

## Data Acquisition

### 1. Automatic Sample Download

To quickly start with a sample dataset (COVID-CT slices), run:

```bash
python download_data.py
```

This will download and extract images to the `data/` directory.

### 2. External Professional Datasets

For full-scale training, consider these high-quality public resources:

* **The Cancer Imaging Archive (TCIA)**: specifically the [LDCT-and-Projection-data](https://wiki.cancerimagingarchive.net/display/Public/LDCT-and-Projection-data) (Mayo Clinic). Contains paired normal/low-dose CTs.
* **CT-ORG**: 140 CT scans with various organ segmentations.
* **COVID-CT Dataset**: Publicly available on Kaggle and GitHub.

**Note**: The framework now supports direct loading of `.dcm` (DICOM) files. Simply place them in the `data/` folder.

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

## Structure

- `data/`: Place your High-Resolution CT images here (PNG, JPG, TIF).
* `model.py`: Defines the SRResNet architecture.
* `dataset.py`: Handles data loading and synthetic downsampling.
* `train.py`: Script to train the model.
* `evaluate.py`: Script to evaluate the model and generate results.

## Usage

### 1. Data Preparation

Place your training images in `data/`. If no images are present, the training script will generate synthetic dummy data for demonstration.

### 2. Training

Run the training script:

```bash
python train.py --epochs 10 --batch_size 4 --scale_factor 4
```

Options:
* `--data_dir`: Path to image directory (default: `./data`)
* `--scale_factor`: Upsampling scale (default: 4)
* `--epochs`: Number of epochs (default: 10)

This will save checkpoints to `checkpoints/` and a training loss curve `training_loss.png`.

### 3. Evaluation

Run the evaluation script:

```bash
python evaluate.py --model_path checkpoints/model_final.pth --scale_factor 4
```

This will:
* Calculate PSNR, SSIM, MSE, and Inference Time.
* Save visual comparisons (LR vs SR vs HR) to `results/`.
* Save metric distribution histograms to `results/metrics_distribution.png`.

## Notes on Metrics

- **AUC**: Area Under Curve is typically used for classification tasks. For Super-Resolution (a regression task), we use **PSNR**, **SSIM**, and **MSE** (Error) to measure reconstruction quality.
* **Error**: Represented by MSE (Mean Squared Error).
