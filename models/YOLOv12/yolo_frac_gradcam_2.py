"""
YOLO Fracture Visualization - Simple & Robust Version
--------------------------------------------------------
Uses feature activation maps (more reliable than Grad-CAM for YOLO)
and creates clean visualizations of attention + bounding box comparisons.

Requirements:
    pip install ultralytics opencv-python matplotlib numpy

Usage:
    1. Place your images and model in the same directory as this script
    2. Update the configuration section below
    3. Run: python yolo_fracture_simple.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
from ultralytics import YOLO

# CONFIGURATION - Update paths

MODEL_PATH = "runs_frac/yolo12n_fracatlas_tuned/weights/best.pt"

# Your X-ray images with annotations
IMAGES = [
    {
        "path": "datasets/fracatlas/images/val/IMG0003924.jpg",
        "gt_yolo": [0, 0.5250, 0.6419, 0.0981, 0.2031],  # class, x_center, y_center, w, h
        "predictions": [
            {"class": 0, "score": 0.759, "box": [177.0, 245.2, 216.7, 335.6]},
            {"class": 0, "score": 0.340, "box": [154.9, 18.4, 169.3, 90.8]},
        ]
    },
    {
        "path": "datasets/fracatlas/images/val/IMG0004208.jpg",
        "gt_yolo": [0, 0.5190, 0.5034, 0.0894, 0.0811],
        "predictions": [
            {"class": 0, "score": 0.769, "box": [178.2, 202.7, 210.6, 249.4]},
        ]
    }
]

OUTPUT_DIR = "fracture_results"

# Helper Functions 

def yolo_to_xyxy(yolo_box, img_w, img_h):
    """Convert YOLO format to pixel coordinates [x1, y1, x2, y2]."""
    _, xc, yc, w, h = yolo_box
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x1, y1, x2, y2]


class YOLOFeatureExtractor:
    """Extract feature maps from YOLO for visualization."""
    
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on backbone layers."""
        torch_model = self.model.model
        
        # Hook into multiple layers for richer visualization
        target_indices = [4, 6, 9]  # Different depths of the backbone
        
        for idx in target_indices:
            if idx < len(torch_model.model):
                layer = torch_model.model[idx]
                hook = layer.register_forward_hook(
                    lambda m, inp, out, idx=idx: self._save_features(idx, out)
                )
                self.hooks.append(hook)
    
    def _save_features(self, idx, output):
        self.features[idx] = output.detach()
    
    def get_attention_map(self, image_path, layer_idx=9):
        """
        Generate attention heatmap from feature activations.
        
        Args:
            image_path: Path to image
            layer_idx: Which layer to use (higher = more semantic)
        
        Returns:
            heatmap: Normalized attention map
        """
        # Clear previous features
        self.features = {}
        
        # Run inference to trigger hooks
        img = cv2.imread(image_path)
        _ = self.model(image_path, verbose=False)
        
        if layer_idx not in self.features:
            # Fall back to available layer
            layer_idx = list(self.features.keys())[-1] if self.features else None
            if layer_idx is None:
                return np.zeros((img.shape[0], img.shape[1]))
        
        # Get feature maps
        feat = self.features[layer_idx].cpu().numpy()[0]  # [C, H, W]
        
        # Method 1: Mean absolute activation
        heatmap = np.mean(np.abs(feat), axis=0)
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def get_multi_scale_attention(self, image_path):
        """
        Combine attention from multiple layers for richer visualization.
        """
        self.features = {}
        img = cv2.imread(image_path)
        _ = self.model(image_path, verbose=False)
        
        combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        for idx, feat_tensor in self.features.items():
            feat = feat_tensor.cpu().numpy()[0]
            layer_map = np.mean(np.abs(feat), axis=0)
            layer_map = cv2.resize(layer_map, (img.shape[1], img.shape[0]))
            layer_map = (layer_map - layer_map.min()) / (layer_map.max() - layer_map.min() + 1e-8)
            combined += layer_map
        
        if len(self.features) > 0:
            combined /= len(self.features)
        
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        return combined
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def create_overlay(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image."""
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1-alpha, heatmap_color, alpha, 0)


def visualize_single_image(img_path, gt_box, predictions, heatmap, output_path):
    """Create visualization for a single image."""
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # fig.suptitle(f"Fracture Detection: {os.path.basename(img_path)}", fontsize=14, y=1.02)
    #fig.suptitle(f"Fracture Detection with YOLO12n", fontsize=24, y=1.02)
    
    
    # Panel 1: Original
    axes[0].imshow(img_rgb, cmap='gray')
    axes[0].set_title("(A) Original X-Ray")
    axes[0].axis('off')
    
    # Panel 2: Ground Truth + Predictions
    axes[1].imshow(img_rgb, cmap='gray')
    
    # Ground truth (green, solid)
    x1, y1, x2, y2 = gt_box
    rect_gt = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor='purple', 
                                  facecolor='none', label='Ground Truth')
    axes[1].add_patch(rect_gt)
    
    # Predictions (red, dashed)
    for i, pred in enumerate(predictions):
        px1, py1, px2, py2 = pred['box']
        rect_pred = patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                        linewidth=2, edgecolor='white',
                                        facecolor='none', linestyle='--',
                                        #label=f"Pred (conf={pred['score']:.2f})" if i==0 else None) ***
                                        label=f"Prediction" if i==0 else None
                                        )
        axes[1].add_patch(rect_pred)
        # Add confidence label
        axes[1].text(px1, py1-5, f"{pred['score']:.2f}", color='red', fontsize=10.5)
    
    axes[1].legend(loc='upper right')       # need this if *** holds: bbox_to_anchor=(0.999, 0.97) 
    axes[1].set_title("(B) Ground Truth vs Predictions")
    axes[1].axis('off')
    
    # Panel 3: Heatmap
    im = axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title("(C) Attention Heatmap")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Panel 4: Overlay with boxes
    overlay = create_overlay(img_bgr, heatmap, alpha=0.45)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[3].imshow(overlay_rgb)
    
    # Draw boxes on overlay
    rect_gt = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=3, edgecolor='purple', facecolor='none')
    axes[3].add_patch(rect_gt)
    
    for pred in predictions:
        px1, py1, px2, py2 = pred['box']
        rect_pred = patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                        linewidth=2, edgecolor='white',
                                        facecolor='none', linestyle='--') 
        axes[3].add_patch(rect_pred)

    axes[3].set_title("(D) Heatmap + Detections")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_combined_figure(results_list, output_path):
    """Create a publication-ready combined figure."""
    
    n = len(results_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, res in enumerate(results_list):
        img_rgb = cv2.cvtColor(cv2.imread(res['path']), cv2.COLOR_BGR2RGB)
        gt_box = res['gt_box']
        preds = res['predictions']
        heatmap = res['heatmap']
        overlay = create_overlay(cv2.imread(res['path']), heatmap, alpha=0.45)
        
        # Column 1: Detections
        axes[i, 0].imshow(img_rgb, cmap='gray')
        x1, y1, x2, y2 = gt_box
        axes[i, 0].add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor='lime', facecolor='none'))
        for p in preds:
            px1, py1, px2, py2 = p['box']
            axes[i, 0].add_patch(patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                 linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
        axes[i, 0].set_ylabel(os.path.basename(res['path']).replace('.jpg',''), fontsize=10)
        axes[i, 0].set_title("Detection" if i == 0 else "")
        axes[i, 0].axis('off')
        
        # Column 2: Heatmap
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Attention" if i == 0 else "")
        axes[i, 1].axis('off')
        
        # Column 3: Overlay
        axes[i, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title("Combined" if i == 0 else "")
        axes[i, 2].axis('off')
    
    # Add legend at bottom
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='lime', linewidth=2, label='Ground Truth'),
        patches.Patch(facecolor='none', edgecolor='red', linewidth=2, linestyle='--', label='Prediction')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved combined figure: {output_path}")

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    print("Initializing feature extractor...")
    extractor = YOLOFeatureExtractor(model)
    
    results_list = []
    
    for img_data in IMAGES:
        img_path = img_data['path']
        print(f"\nProcessing: {img_path}")
        
        # Load image dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ERROR: Could not load {img_path}")
            continue
        
        h, w = img.shape[:2]
        
        # Convert GT to pixels
        gt_box = yolo_to_xyxy(img_data['gt_yolo'], w, h)
        print(f"  GT box: [{gt_box[0]:.1f}, {gt_box[1]:.1f}, {gt_box[2]:.1f}, {gt_box[3]:.1f}]")
        
        # Generate attention heatmap
        print("  Generating attention map...")
        heatmap = extractor.get_multi_scale_attention(img_path)
        
        # Save individual visualization
        out_path = os.path.join(OUTPUT_DIR, f"{Path(img_path).stem}_analysis.png")
        visualize_single_image(img_path, gt_box, img_data['predictions'], heatmap, out_path)
        
        # Store for combined figure
        results_list.append({
            'path': img_path,
            'gt_box': gt_box,
            'predictions': img_data['predictions'],
            'heatmap': heatmap
        })
    
    # Create combined figure
    if results_list:
        combined_path = os.path.join(OUTPUT_DIR, "combined_analysis.png")
        create_combined_figure(results_list, combined_path)
    
    extractor.cleanup()
    
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
