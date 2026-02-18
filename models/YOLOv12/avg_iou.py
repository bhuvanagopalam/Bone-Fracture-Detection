"""
Compute Average IoU for YOLO12 on Test Set
------------------------------------------
Calculates the mean IoU between predicted and ground truth bounding boxes
for all true positive detections.

Requirements:
    pip install ultralytics numpy

Usage:
    python compute_avg_iou.py
"""

from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Configuration - Update these paths

MODEL_PATH = "runs_frac/yolo12n_fracatlas_tuned/weights/best.pt"
TEST_IMAGES_PATH = "datasets/fracatlas/images/test"  # Direct path to test images
TEST_LABELS_PATH = "datasets/fracatlas/labels/test"  # Direct path to test labels
CONF_THRESHOLD = 0.5  # Same threshold used for evaluation
IOU_THRESHOLD = 0.5   # IoU threshold to consider a detection as true positive
DEVICE = 0  # GPU (use 'cpu' if no GPU)

# Helper Function - IoU Computation

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    Boxes are in format [x1, y1, x2, y2] (xyxy).
    """
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Compute IoU
    if union == 0:
        return 0.0
    return intersection / union

# Main

def main():
    print("=" * 50)
    print("  AVERAGE IoU COMPUTATION")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Get test images
    test_images_dir = Path(TEST_IMAGES_PATH)
    test_labels_dir = Path(TEST_LABELS_PATH)
    
    print(f"Test images path: {test_images_dir}")
    print(f"Test labels path: {test_labels_dir}")
    
    # Get all test images
    test_images = list(test_images_dir.glob('*.jpg'))
    
    print(f"Found {len(test_images)} test images")
    
    # Storage for IoU values
    all_ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for img_path in test_images:
        # Get ground truth label file
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        
        # Load ground truth boxes
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height (normalized)
                        cls, xc, yc, w, h = map(float, parts[:5])
                        gt_boxes.append([cls, xc, yc, w, h])
        
        # Run inference
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        
        # Get predictions in xyxy format (pixels)
        pred_boxes = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                pred_boxes.append({
                    'box': xyxy,
                    'conf': conf,
                    'cls': cls
                })
        
        # Convert GT boxes to xyxy pixels
        img_h, img_w = result.orig_shape
        gt_boxes_xyxy = []
        for gt in gt_boxes:
            cls, xc, yc, w, h = gt
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            gt_boxes_xyxy.append([x1, y1, x2, y2])
        
        # Match predictions to ground truths
        matched_gt = set()
        
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes_xyxy):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred['box'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= IOU_THRESHOLD and best_gt_idx >= 0:
                # True positive
                all_ious.append(best_iou)
                matched_gt.add(best_gt_idx)
                true_positives += 1
            else:
                # False positive
                false_positives += 1
        
        # Count unmatched ground truths as false negatives
        false_negatives += len(gt_boxes_xyxy) - len(matched_gt)
    
    # Compute statistics
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    
    if len(all_ious) > 0:
        avg_iou = np.mean(all_ious)
        std_iou = np.std(all_ious)
        min_iou = np.min(all_ious)
        max_iou = np.max(all_ious)
        median_iou = np.median(all_ious)
        
        print(f"\nTrue Positives:  {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"\nIoU Statistics (over {len(all_ious)} true positive detections):")
        print(f"  Average IoU: {avg_iou:.4f}")
        print(f"  Std Dev:     {std_iou:.4f}")
        print(f"  Median IoU:  {median_iou:.4f}")
        print(f"  Min IoU:     {min_iou:.4f}")
        print(f"  Max IoU:     {max_iou:.4f}")
        
        # Also compute precision and recall for reference
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"\nFor reference:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
    else:
        print("\nNo true positive detections found!")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
    
    return avg_iou if len(all_ious) > 0 else 0.0


if __name__ == "__main__":
    avg_iou = main()
