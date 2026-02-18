"""
This script obtains the predicted bounding boxes for any X-ray in the test, validation (val), or train set.
----------------------------------------------------------------------------------------------------------
You have to load the trained weights you want to use (suggest using the best weights obtained from training).

Requirements:
    pathlib
    numpy
    ultralytics YOLO

"""

from pathlib import Path
import numpy as np
from ultralytics import YOLO

# 1) Load your tuned nano model - Update this path
WEIGHTS_PATH = "runs_frac/yolo12n_fracatlas_tuned/weights/best.pt"
model = YOLO(WEIGHTS_PATH)

# 2) Paths to the two X-rays you care about - Update these paths
DATA_ROOT = Path("datasets/fracatlas")
image_paths = [
    DATA_ROOT / "images" / "val" / "IMG0003924.jpg",
    DATA_ROOT / "images" / "val" / "IMG0004208.jpg",
]

for img_path in image_paths:
    print(f"\n=== Predictions for {img_path.name} ===")

    # Run inference
    results = model(
        source=str(img_path),
        imgsz=640,     # same as training
        conf=0.25,     # confidence threshold
        iou=0.7,       # NMS IoU threshold (optional)
        device=0,      # GPU, use 'cpu' if needed
        verbose=False,
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        print("No detections.")
        continue

    # Boxes in xyxy pixels, confidences, and class indices
    boxes_xyxy = results.boxes.xyxy.cpu().numpy()   # shape [N, 4]
    scores     = results.boxes.conf.cpu().numpy()   # shape [N]
    classes    = results.boxes.cls.cpu().numpy()    # shape [N] (should all be 0 for 'fracture')

    for i, (box, score, cls) in enumerate(zip(boxes_xyxy, scores, classes)):
        x1, y1, x2, y2 = box
        print(f"Detection {i}:")
        print(f"  class = {int(cls)}  (fracture)")
        print(f"  score = {score:.3f}")
        print(f"  box   = [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
