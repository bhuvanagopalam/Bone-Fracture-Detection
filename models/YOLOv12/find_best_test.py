"""
Find Best IoU Examples from Test Set
--------------------------------------
Identifies test images with highest IoU between predictions and ground truth,
and outputs the coordinates needed for visualization.

Usage:
    python find_best_iou_examples.py
"""

from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Configuration - Update these paths

MODEL_PATH = "runs_frac/yolo12n_fracatlas_tuned/weights/best.pt"
TEST_IMAGES_PATH = "datasets/fracatlas/images/test"
TEST_LABELS_PATH = "datasets/fracatlas/labels/test"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEVICE = 0
NUM_EXAMPLES = 5  # How many top examples to show

# Helper Function

def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union

# Main

def main():
    print("=" * 60)
    print("  FINDING BEST IoU EXAMPLES FOR VISUALIZATION")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Get test images
    test_images_dir = Path(TEST_IMAGES_PATH)
    test_labels_dir = Path(TEST_LABELS_PATH)
    test_images = list(test_images_dir.glob('*.jpg'))
    
    print(f"Found {len(test_images)} test images\n")
    
    # Store all detections with their IoU
    all_detections = []
    
    for img_path in test_images:
        # Get ground truth
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        
        gt_boxes_yolo = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, w, h = map(float, parts[:5])
                        gt_boxes_yolo.append([cls, xc, yc, w, h])
        
        if not gt_boxes_yolo:
            continue
        
        # Run inference
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        
        if len(result.boxes) == 0:
            continue
        
        # Get image dimensions
        img_h, img_w = result.orig_shape
        
        # Convert GT to xyxy pixels
        gt_boxes_xyxy = []
        for gt in gt_boxes_yolo:
            cls, xc, yc, w, h = gt
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            gt_boxes_xyxy.append({
                'yolo': gt,
                'xyxy': [x1, y1, x2, y2]
            })
        
        # Get predictions
        for box in result.boxes:
            pred_xyxy = box.xyxy[0].cpu().numpy().tolist()
            pred_conf = float(box.conf[0].cpu().numpy())
            
            # Find best matching GT
            best_iou = 0
            best_gt = None
            
            for gt in gt_boxes_xyxy:
                iou = compute_iou(pred_xyxy, gt['xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou >= IOU_THRESHOLD and best_gt is not None:
                all_detections.append({
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'iou': best_iou,
                    'pred_conf': pred_conf,
                    'pred_xyxy': pred_xyxy,
                    'gt_yolo': best_gt['yolo'],
                    'gt_xyxy': best_gt['xyxy'],
                    'img_size': (img_w, img_h)
                })
    
    # Sort by IoU (highest first)
    all_detections.sort(key=lambda x: x['iou'], reverse=True)
    
    # Print top examples
    print("=" * 60)
    print(f"  TOP {NUM_EXAMPLES} EXAMPLES BY IoU")
    print("=" * 60)
    
    for i, det in enumerate(all_detections[:NUM_EXAMPLES]):
        print(f"\n{'='*60}")
        print(f"RANK {i+1}: {det['image_name']}")
        print(f"{'='*60}")
        print(f"  IoU:        {det['iou']:.4f}")
        print(f"  Confidence: {det['pred_conf']:.3f}")
        print(f"  Image size: {det['img_size'][0]} x {det['img_size'][1]}")
        print(f"\n  Ground Truth (YOLO format):")
        print(f"    [class, x_center, y_center, width, height]")
        print(f"    {det['gt_yolo']}")
        print(f"\n  Ground Truth (xyxy pixels):")
        print(f"    [{det['gt_xyxy'][0]:.1f}, {det['gt_xyxy'][1]:.1f}, {det['gt_xyxy'][2]:.1f}, {det['gt_xyxy'][3]:.1f}]")
        print(f"\n  Prediction (xyxy pixels):")
        print(f"    [{det['pred_xyxy'][0]:.1f}, {det['pred_xyxy'][1]:.1f}, {det['pred_xyxy'][2]:.1f}, {det['pred_xyxy'][3]:.1f}]")
    
    # Print copy-paste ready format for visualization script
    print("\n" + "=" * 60)
    print("  COPY-PASTE FORMAT FOR VISUALIZATION SCRIPT")
    print("=" * 60)
    
    for i, det in enumerate(all_detections[:3]):  # Top 3 for the combined figure, in format for easy copy and paste for attention map visualization code
        print(f"""
    {{
        "path": "datasets/fracatlas/images/test/{det['image_name']}",
        "gt_yolo": {det['gt_yolo']},
        "predictions": [
            {{"class": 0, "score": {det['pred_conf']:.3f}, "box": [{det['pred_xyxy'][0]:.1f}, {det['pred_xyxy'][1]:.1f}, {det['pred_xyxy'][2]:.1f}, {det['pred_xyxy'][3]:.1f}]}},
        ]
    }},""")


if __name__ == "__main__":
    main()
