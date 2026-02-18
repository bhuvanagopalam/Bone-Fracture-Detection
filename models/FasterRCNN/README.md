## Faster R-CNN Model Code

## Dependencies

A complete list of dependencies required for this project is provided in the included **`requirements.txt`** file.

To install all necessary packages, run:

```bash
pip install -r requirements.txt
```



This project implements Faster R-CNN from scratch using PyTorch components, applied to bone fracture detection on the FracAtlas dataset.
The goal is to deeply understand each step of the pipeline:

	•	Custom dataset loader following COCO-style annotations
	
	•	Manual backbone construction using ResNet-50 (C5 feature map)
	
	•	Manual anchor generation for the Region Proposal Network (RPN)

	•	Manual ROI pooling using MultiScaleRoIAlign
	
	•	Custom training loop with per-step and per-epoch logging
	
	•	Custom evaluator computing:
	
	•	mAP@50
	
	•	mAP@50–95
	
	•	Mean Average Recall (MAR)
	
	•	Mean IoU
	
	•	Visualization utilities for displaying predictions with ground-truth overlays

This implementation mirrors the original Faster R-CNN architecture but avoids using the high-level torchvision.models.detection.FasterRCNN wrapper, allowing full visibility and control over internal components.

A second notebook (provided separately) includes the standard torchvision Faster R-CNN model for performance comparison.
