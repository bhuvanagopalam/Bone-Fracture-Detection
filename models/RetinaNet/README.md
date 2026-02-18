# RetinaNet for Bone Fracture Detection


## Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This project uses the FracAtlas dataset with PASCAL VOC format annotations.

### Dataset Structure

Ensure your dataset is organized as follows:

```
FracAtlas/
├── Annotations/
│   └── PASCAL VOC/
│       ├── image1.xml
│       ├── image2.xml
│       └── ...
└── all_images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Preparing the Data

1. **Convert annotations and create dataset splits:**

   Modify  `data_prepration.py` to point to your dataset location:

2. **Run the data preparation script:**

   ```bash
   python data_prepration.py
   ```

   This will generate:
   - `data.csv` - Complete dataset annotations
   - `train_filtered.csv` - Training set
   - `val_filtered.csv` - Validation set
   - `test_filtered.csv` - Test set


## Training

To train RetinaNet run:

```bash
python train_retina.py
```

### Training Configuration

The default training parameters in `train_retina.py` are:

- **Batch Size:** 8
- **Learning Rate:** 0.0001
- **Optimizer:** AdamW with weight decay of 1e-5
- **Scheduler:** CosineAnnealingLR
- **Epochs:** 100
- **Number of Classes:** 2 (fractured, non-fractured)
- **Workers:** 16
- **Pretrained:** True


### Training Output

During training, the script will:
- Save the best model checkpoint as `retinanet_pretrained_{IoU}.pt`
- Generate `train_loss_pretrained.txt` with training losses
- Generate `val_iou_pretrained.txt` with validation IoU scores
- Display validation IoU every 10 epochs




## Reference

[1] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).


