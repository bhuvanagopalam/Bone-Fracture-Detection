"""
Find False Negative Examples from Test Set
-------------------------------------------
Identifies test images where the model failed to detect a fracture
(ground truth exists but no prediction was made, or prediction was poor).

Categorizes false negatives by:
- Complete miss (no predictions at all)
- Poor localization (predictions exist but low IoU)
- Near miss (predictions exist with moderate IoU, just under threshold)

Usage:
    python find_false_negatives.py
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os

# Configuration - Update these paths

MODEL_PATH = "runs_frac/yolo12n_fracatlas_tuned/weights/best.pt"
TEST_IMAGES_PATH = "datasets/fracatlas/images/test"
TEST_LABELS_PATH = "datasets/fracatlas/labels/test"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEVICE = 0
NUM_EXAMPLES = 5
OUTPUT_DIR = "false_negative_results"

# Helper Functions

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


def yolo_to_xyxy(yolo_box, img_w, img_h):
    """Convert YOLO format to pixel coordinates [x1, y1, x2, y2]."""
    cls, xc, yc, w, h = yolo_box
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x1, y1, x2, y2]


def visualize_false_negative(img_path, gt_boxes_xyxy, pred_boxes_info, best_iou, output_path):
    """
    Visualize a false negative case - showing GT box and any predictions made.
    """
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.imshow(img_rgb, cmap='gray')
    
    # Draw ground truth boxes (purple)
    for i, gt_box in enumerate(gt_boxes_xyxy):
        x1, y1, x2, y2 = gt_box
        rect_gt = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=3, edgecolor='purple',
            facecolor='none', linestyle='-',
            label='Ground Truth' if i == 0 else None
        )
        ax.add_patch(rect_gt)
    
    # Draw prediction boxes if any (white dashed)
    for i, pred_info in enumerate(pred_boxes_info):
        px1, py1, px2, py2 = pred_info['box']
        rect_pred = patches.Rectangle(
            (px1, py1), px2-px1, py2-py1,
            linewidth=2, edgecolor='white',
            facecolor='none', linestyle='--',
            label=f'Prediction (conf={pred_info["conf"]:.2f})' if i == 0 else None
        )
        ax.add_patch(rect_pred)
    
    ax.legend(loc='upper right')
    
    # Title with IoU info
    if best_iou == 0 and len(pred_boxes_info) == 0:
        #title = f"Complete Miss: {img_path.name}\n(No predictions made)"
        title = f"No prediction(s) made"
    elif best_iou == 0:
        title = f"False Negative: {img_path.name}\n(Predictions made but no overlap with GT)"
    else:
        title = f"False Negative: {img_path.name}\n(Best IoU with GT: {best_iou:.2f})"
    
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

# Main

def main():
    print("=" * 60)
    print("  FINDING FALSE NEGATIVE EXAMPLES")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Get test images
    test_images_dir = Path(TEST_IMAGES_PATH)
    test_labels_dir = Path(TEST_LABELS_PATH)
    test_images = list(test_images_dir.glob('*.jpg'))
    
    print(f"Found {len(test_images)} test images\n")
    
    # Store false negatives by category
    complete_misses = []      # No predictions at all
    poor_localizations = []   # Predictions exist but IoU < 0.2
    near_misses = []          # Predictions exist with 0.2 <= IoU < 0.5
    
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
        
        # Skip images with no ground truth
        if not gt_boxes_yolo:
            continue
        
        # Run inference
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        
        # Get image dimensions
        img_h, img_w = result.orig_shape
        
        # Convert GT to xyxy pixels
        gt_boxes_xyxy = []
        for gt in gt_boxes_yolo:
            gt_boxes_xyxy.append(yolo_to_xyxy(gt, img_w, img_h))
        
        # Get predictions with confidence
        pred_boxes_info = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                pred_xyxy = box.xyxy[0].cpu().numpy().tolist()
                pred_conf = float(box.conf[0].cpu().numpy())
                pred_boxes_info.append({'box': pred_xyxy, 'conf': pred_conf})
        
        # Check each GT box - is it a false negative?
        for gt_idx, gt_box in enumerate(gt_boxes_xyxy):
            best_iou = 0
            
            for pred_info in pred_boxes_info:
                iou = compute_iou(gt_box, pred_info['box'])
                if iou > best_iou:
                    best_iou = iou
            
            if best_iou < IOU_THRESHOLD:
                # This GT was not properly detected - it's a false negative
                fn_entry = {
                    'image_path': img_path,
                    'image_name': img_path.name,
                    'gt_yolo': gt_boxes_yolo[gt_idx],
                    'gt_xyxy': gt_box,
                    'all_gt_xyxy': gt_boxes_xyxy,
                    'pred_boxes_info': pred_boxes_info,
                    'best_iou': best_iou,
                    'img_size': (img_w, img_h),
                    'num_predictions': len(pred_boxes_info)
                }
                
                # Categorize
                if len(pred_boxes_info) == 0:
                    complete_misses.append(fn_entry)
                elif best_iou < 0.2:
                    poor_localizations.append(fn_entry)
                else:
                    near_misses.append(fn_entry)
    
    # Sort each category by best_iou (sort lowest first for complete misses and poor predictions; sort highest first for near misses)
    complete_misses.sort(key=lambda x: x['best_iou'])
    poor_localizations.sort(key=lambda x: x['best_iou'])
    near_misses.sort(key=lambda x: x['best_iou'], reverse=True)
    
    # Print summary
    print("=" * 60)
    print("  FALSE NEGATIVE SUMMARY")
    print("=" * 60)
    print(f"\n  Complete misses (no predictions):     {len(complete_misses)}")
    print(f"  Poor localizations (IoU < 0.2):       {len(poor_localizations)}")
    print(f"  Near misses (0.2 <= IoU < 0.5):       {len(near_misses)}")
    print(f"  Total false negatives:                {len(complete_misses) + len(poor_localizations) + len(near_misses)}")
    
    # Print and visualize examples from each category
    visualized_count = 0
    
    # Priority: Complete misses first, then poor localizations
    print("\n" + "=" * 60)
    print("  COMPLETE MISSES (Best for showing failure)")
    print("=" * 60)
    
    for i, fn in enumerate(complete_misses[:NUM_EXAMPLES]):
        print(f"\n{'='*60}")
        print(f"COMPLETE MISS {i+1}: {fn['image_name']}")
        print(f"{'='*60}")
        print(f"  Image size: {fn['img_size'][0]} x {fn['img_size'][1]}")
        print(f"  Predictions made: {fn['num_predictions']}")
        print(f"  Best IoU with GT: {fn['best_iou']:.2f}")
        print(f"\n  Ground Truth (YOLO format):")
        print(f"    {fn['gt_yolo']}")
        print(f"\n  Ground Truth (xyxy pixels):")
        print(f"    [{fn['gt_xyxy'][0]:.1f}, {fn['gt_xyxy'][1]:.1f}, {fn['gt_xyxy'][2]:.1f}, {fn['gt_xyxy'][3]:.1f}]")
        
        # Visualize
        output_path = os.path.join(OUTPUT_DIR, f"complete_miss_{i+1}_{fn['image_name'].replace('.jpg', '.png')}")
        visualize_false_negative(fn['image_path'], fn['all_gt_xyxy'], fn['pred_boxes_info'], fn['best_iou'], output_path)
        visualized_count += 1
    
    if len(poor_localizations) > 0:
        print("\n" + "=" * 60)
        print("  POOR LOCALIZATIONS (IoU < 0.2)")
        print("=" * 60)
        
        for i, fn in enumerate(poor_localizations[:max(0, NUM_EXAMPLES - len(complete_misses))]):
            print(f"\n{'='*60}")
            print(f"POOR LOCALIZATION {i+1}: {fn['image_name']}")
            print(f"{'='*60}")
            print(f"  Image size: {fn['img_size'][0]} x {fn['img_size'][1]}")
            print(f"  Predictions made: {fn['num_predictions']}")
            print(f"  Best IoU with GT: {fn['best_iou']:.2f}")
            print(f"\n  Ground Truth (YOLO format):")
            print(f"    {fn['gt_yolo']}")
            
            # Visualize
            output_path = os.path.join(OUTPUT_DIR, f"poor_localization_{i+1}_{fn['image_name'].replace('.jpg', '.png')}")
            visualize_false_negative(fn['image_path'], fn['all_gt_xyxy'], fn['pred_boxes_info'], fn['best_iou'], output_path)
            visualized_count += 1
    
    if len(near_misses) > 0:
        print("\n" + "=" * 60)
        print("  NEAR MISSES (0.2 <= IoU < 0.5) - NOT recommended for failure case")
        print("=" * 60)
        
        for i, fn in enumerate(near_misses[:3]):  # Just show a few for reference
            print(f"\n  Near miss: {fn['image_name']} (IoU: {fn['best_iou']:.2f})")
    
    print(f"\n{'='*60}")
    print(f"  Visualizations saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
