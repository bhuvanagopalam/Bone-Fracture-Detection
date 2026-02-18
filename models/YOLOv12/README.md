# YOLOv12 Fracture Detection

This folder contains all scripts for training, evaluating, and visualizing a YOLOv12-nano model fine-tuned on the FracAtlas dataset for bone fracture detection.

## Table of Contents
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [File Descriptions](#file-descriptions)
- [Path Configuration](#path-configuration)
- [Usage Guide](#usage-guide)
- [Expected Outputs](#expected-outputs)

---

## Requirements

### Python Version
- Python 3.8+

### Dependencies
Install all required packages:
```bash
pip install ultralytics opencv-python matplotlib numpy pandas torch
```

### Package Versions (Tested)
| Package | Version |
|---------|---------|
| ultralytics | 8.0+ |
| opencv-python | 4.8+ |
| matplotlib | 3.7+ |
| numpy | 1.24+ |
| pandas | 2.0+ |
| torch | 2.0+ |

### Hardware
- **GPU recommended** for training and inference (CUDA-compatible)
- Scripts default to `device=0` (first GPU). Change to `device='cpu'` if no GPU is available.

---

## Dataset Setup

### 1. Download FracAtlas Dataset
Download the FracAtlas dataset from [Figshare](https://figshare.com/articles/dataset/The_dataset/22363012).

### 2. Prepare Dataset for YOLO Format
Run the preparation script to convert FracAtlas to YOLO format:
```bash
python prepare_fracatlas_for_yolo.py
```

This creates the following structure:
```
datasets/
└── fracatlas/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

### 3. Verify Configuration
Ensure `fracatlas.yaml` points to the correct dataset location:
```yaml
path: datasets/fracatlas
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: fracture
```

---

## File Descriptions

### Data Preparation
| File | Description |
|------|-------------|
| `prepare_fracatlas_for_yolo.py` | Converts raw FracAtlas dataset to YOLO format by organizing images and labels into train/val/test splits |
| `fracatlas.yaml` | YOLO dataset configuration file specifying paths and class names |

### Training & Evaluation
| File | Description |
|------|-------------|
| `yolo_frac_tuned.py` | Main training script. Fine-tunes YOLOv12-nano on FracAtlas with tuned hyperparameters (lr0=0.0015, lrf=0.01, 500 epochs) |
| `yolo_frac_eval.py` | Evaluation script. Runs the trained model on the test set and reports precision, recall, and mAP@0.5 |
| `avg_iou.py` | Computes average IoU between predicted and ground truth bounding boxes for all true positive detections |

### Analysis & Visualization
| File | Description |
|------|-------------|
| `get_results.py` | Quick inference script to get prediction coordinates for specific images |
| `find_best_test.py` | Finds test images with highest IoU scores for visualization examples |
| `find_false_negatives.py` | Identifies and visualizes false negative cases (missed detections) |
| `yolo_frac_gradcam_2.py` | Generates attention heatmaps and detection visualizations using feature activation maps |
| `fracture_demo.py` | Interactive demo comparing human vs. AI fracture detection |
| `fracture_distribution.py` | Generates bar charts showing fracture distribution by body region |

---

## Path Configuration

Most scripts require path updates in their `CONFIGURATION` section. Below are **some** of the key paths to modify (it is up to the user to ensure all necessary paths are correct in each script):

### Common Paths
```python
# Model weights (after training)
MODEL_PATH = "runs_frac/yolo12n_fracatlas_tuned/weights/best.pt"

# Dataset paths
TEST_IMAGES_PATH = "datasets/fracatlas/images/test"
TEST_LABELS_PATH = "datasets/fracatlas/labels/test"

# Dataset YAML
data_yaml = "fracatlas.yaml"
```

### Some Key Script-Specific Paths (refer to each *.py script in this folder for details) 

| Script | Paths to Update |
|--------|-----------------|
| `prepare_fracatlas_for_yolo.py` | `FRAC_ROOT` - path to downloaded FracAtlas folder |
| `fracture_distribution.py` | `CSV_PATH` - path to `FracAtlas/dataset.csv` |
| `fracture_demo.py` | `IMAGE_PATH`, `OVERLAY_PATH` - paths to demo image and pre-generated heatmap overlay |
| `yolo_frac_gradcam_2.py` | `IMAGES` list - update image paths and ground truth coordinates |

### Device Configuration
All scripts default to GPU (`device=0`). To use CPU:
```python
DEVICE = 'cpu'  # or device = 'cpu'
```

---

## Usage Guide

### Step 1: Prepare the Dataset
```bash
python prepare_fracatlas_for_yolo.py
```
**Output:** Creates `datasets/fracatlas/` with YOLO-formatted images and labels.

### Step 2: Train the Model
```bash
python yolo_frac_tuned.py
```
**Output:** 
- Trained weights saved to `runs_frac/yolo12n_fracatlas_tuned/weights/best.pt`
- Training metrics and plots in the run directory

**Training Parameters:**
- Base model: `yolo12n.pt`
- Epochs: 500 (with early stopping patience=50)
- Image size: 640
- Batch size: 16
- Learning rate: lr0=0.0015, lrf=0.01

### Step 3: Evaluate on Test Set
```bash
python yolo_frac_eval.py
```
**Output:** Prints precision, recall, and mAP@0.5 on the test set.

### Step 4: Compute Average IoU
```bash
python avg_iou.py
```
**Output:** Prints IoU statistics (mean, std, min, max) over true positive detections.

### Step 5: Generate Visualizations

#### Attention Heatmaps
```bash
python yolo_frac_gradcam_2.py
```
**Output:** Saves visualization figures to `fracture_results/`

#### Find Best Detection Examples
```bash
python find_best_test.py
```
**Output:** Prints top IoU examples with coordinates ready for visualization scripts.

#### Find False Negatives
```bash
python find_false_negatives.py
```
**Output:** Saves false negative visualizations to `false_negative_results/`

#### Dataset Distribution Chart
```bash
python fracture_distribution.py
```
**Output:** Saves `fracture_distribution.png` showing fractures by body region.

### Step 6: Run Interactive Demo
```bash
python fracture_demo.py
```
**Prerequisites:** 
- Run `yolo_frac_gradcam_2.py` first to generate the heatmap overlay (`.npy` file)
- Update `IMAGE_PATH` and `OVERLAY_PATH` in the script

**Instructions:**
1. Click on the X-ray where you think the fracture is
2. Press "Next" to see the AI's prediction
3. Continue pressing "Next" to see heatmap and ground truth comparison

---

## Expected Outputs

### Training Results
After training completes, you should see:
```
runs_frac/
└── yolo12n_fracatlas_tuned/
    ├── weights/
    │   ├── best.pt      # Best model weights
    │   └── last.pt      # Final epoch weights
    ├── results.png      # Training curves
    └── confusion_matrix.png
```

### Evaluation Metrics (Test Set, conf=0.5)
| Metric | Value |
|--------|-------|
| Precision | 0.775 |
| Recall | 0.463 |
| mAP@0.5 | 0.617 |
| Average IoU | 0.711 |

### Visualization Outputs
```
fracture_results/
├── IMG*_analysis.png      # Individual image analysis (4-panel figure)
├── IMG*_overlay.npy       # Saved heatmap overlay for demo
└── combined_analysis.png  # Multi-image comparison figure

false_negative_results/
└── complete_miss_*.png    # False negative visualizations
```

---

## Notes

- **Confidence Threshold:** Evaluation uses `conf=0.5` to filter low-confidence predictions. This can be adjusted in the respective scripts.
- **IoU Threshold:** Default is 0.5 for matching predictions to ground truth.
- **Pre-trained Weights:** The `best.pt` file in this repository contains our fine-tuned weights. To retrain from scratch, run `yolo_frac_tuned.py`.

---

## References

### YOLOv12 Paper
> Y. Tian, Q. Ye, and D. Doermann, "YOLO12: Attention-Centric Real-Time Object Detectors," *arXiv preprint arXiv:2502.12524*, 2025.
```bibtex
@article{tian2025yolo12,
  title={YOLO12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

### YOLOv12 Software
> Y. Tian, Q. Ye, and D. Doermann, "YOLO12: Attention-Centric Real-Time Object Detectors," 2025. Available: https://github.com/sunsmarterjie/yolov12
```bibtex
@software{yolo12,
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  title = {YOLO12: Attention-Centric Real-Time Object Detectors},
  year = {2025},
  url = {https://github.com/sunsmarterjie/yolov12},
  license = {AGPL-3.0}
}
```
