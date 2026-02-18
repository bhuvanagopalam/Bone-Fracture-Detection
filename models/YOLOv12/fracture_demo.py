"""
Interactive Fracture Detection Demo
-------------------------------------
A step-by-step demo comparing human vs. AI fracture detection.

Requirements:
    pip install opencv-python matplotlib numpy

Usage:
    python fracture_demo.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import os

# Configuration - Update these paths

IMAGE_PATH = "datasets/fracatlas/images/val/IMG0004208.jpg"
OVERLAY_PATH = "fracture_results/IMG0004208_overlay.npy"  # Pre-saved heatmap overlay

# Ground truth (YOLO format: class, x_center, y_center, width, height - normalized) - entered manually
GT_YOLO = [0, 0.5190, 0.5034, 0.0894, 0.0811]

# Prediction (xyxy pixels) - entered manually
PREDICTION = {"class": 0, "score": 0.769, "box": [178.2, 202.7, 210.6, 249.4]}

# Helper Functions

def yolo_to_xyxy(yolo_box, img_w, img_h):
    """Convert YOLO format to pixel coordinates [x1, y1, x2, y2]."""
    _, xc, yc, w, h = yolo_box
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x1, y1, x2, y2]


def point_in_box(x, y, box):
    """Check if point (x, y) is inside box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# Interactive Demo

class FractureDemo:
    def __init__(self, image_path, overlay_path, gt_yolo, prediction):
        self.image_path = image_path
        self.overlay_path = overlay_path
        self.gt_yolo = gt_yolo
        self.prediction = prediction
        
        # Load image
        self.img_bgr = cv2.imread(image_path)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = self.img_bgr.shape[:2]
        
        # Convert GT to pixels
        self.gt_box = yolo_to_xyxy(gt_yolo, self.img_w, self.img_h)
        
        # User click storage
        self.user_click = None
        
        # Current step
        self.step = 0
        
        # Pre-saved overlay
        self.overlay_rgb = None
        
        # Figure and axes
        self.fig = None
        self.ax = None
        self.btn_next = None
        
    def load_overlay(self):
        """Load pre-saved heatmap overlay."""
        if self.overlay_rgb is not None:
            return
        
        if os.path.exists(self.overlay_path):
            self.overlay_rgb = np.load(self.overlay_path)
            print(f"Loaded overlay from {self.overlay_path}")
        else:
            print(f"Error: Could not find overlay at {self.overlay_path}")
            # Fallback to original image if overlay not found
            self.overlay_rgb = self.img_rgb
    
    def on_click(self, event):
        """Handle mouse click on image."""
        if self.step != 0:  # Only accept clicks on step 0
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        self.user_click = (event.xdata, event.ydata)
        
        # Show the click as a dot
        self.ax.plot(event.xdata, event.ydata, 'yo', markersize=8, 
                     markeredgecolor='black', markeredgewidth=1, label='Your guess')
        self.ax.legend(loc='upper left')
        self.fig.canvas.draw()
        
        print(f"Click recorded at: ({event.xdata:.1f}, {event.ydata:.1f})")
    
    def on_next(self, event):
        """Handle Next button click."""
        if self.step == 0 and self.user_click is None:
            print("Please click on the image first to mark where you think the fracture is!")
            return
        
        self.step += 1
        self.update_display()
    
    def update_display(self):
        """Update display based on current step."""
        self.ax.clear()
        
        if self.step == 0:
            # Step 0: Show original image, wait for click
            self.ax.imshow(self.img_rgb, cmap='gray')
            self.ax.set_title("Where is the fracture? Click on the image.", fontsize=14)
            self.ax.axis('off')
            
        elif self.step == 1:
            # Step 1: Show image with user click + model prediction
            self.ax.imshow(self.img_rgb, cmap='gray')
            
            # User click
            if self.user_click:
                self.ax.plot(self.user_click[0], self.user_click[1], 'yo', markersize=8,
                            markeredgecolor='black', markeredgewidth=1, label='Your guess')
            
            # Model prediction
            px1, py1, px2, py2 = self.prediction['box']
            pred_rect = patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                          linewidth=3, edgecolor='white', 
                                          facecolor='none', linestyle='--',
                                          label=f"AI prediction (conf={self.prediction['score']:.2f})")
            self.ax.add_patch(pred_rect)
            
            self.ax.legend(loc='upper left')
            self.ax.set_title("Here's what the AI detected!", fontsize=14)
            self.ax.axis('off')
            
        elif self.step == 2:
            # Step 2: Show heatmap overlay
            self.load_overlay()
            
            self.ax.imshow(self.overlay_rgb)
            
            # User click
            if self.user_click:
                self.ax.plot(self.user_click[0], self.user_click[1], 'yo', markersize=8,
                            markeredgecolor='black', markeredgewidth=1, label='Your guess')
            
            self.ax.legend(loc='upper left')
            self.ax.set_title("Did you look in the same place(s) as the AI to inform your guess?\n(Warm colors = where the AI focused)", fontsize=14)
            self.ax.axis('off')
            
        elif self.step == 3:
            # Step 3: Show everything with ground truth
            self.ax.imshow(self.img_rgb, cmap='gray')
            
            # Ground truth
            gx1, gy1, gx2, gy2 = self.gt_box
            gt_rect = patches.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1,
                                        linewidth=3, edgecolor='purple',
                                        facecolor='none', linestyle='-',
                                        label='Ground truth')
            self.ax.add_patch(gt_rect)
            
            # Model prediction
            px1, py1, px2, py2 = self.prediction['box']
            pred_rect = patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                          linewidth=2, edgecolor='white',
                                          facecolor='none', linestyle='--',
                                          label=f"AI prediction")
            self.ax.add_patch(pred_rect)
            
            # User click
            if self.user_click:
                self.ax.plot(self.user_click[0], self.user_click[1], 'yo', markersize=8,
                            markeredgecolor='black', markeredgewidth=1, label='Your guess')
            
            # Check if user was correct
            user_correct = point_in_box(self.user_click[0], self.user_click[1], self.gt_box) if self.user_click else False
            
            if user_correct:
                title = "Congrats! You found the fracture!"
                title_color = 'green'
            else:
                title = "Good try!\n The fracture was actually located in the purple bounding box!"
                title_color = 'purple'
            
            self.ax.legend(loc='upper left')
            self.ax.set_title(title, fontsize=14, color=title_color, fontweight='bold')
            self.ax.axis('off')
            
        elif self.step == 4:
            # Step 4: Final summary with heatmap overlay + all boxes
            self.load_overlay()
            
            self.ax.imshow(self.overlay_rgb)
            
            # Ground truth
            gx1, gy1, gx2, gy2 = self.gt_box
            gt_rect = patches.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1,
                                        linewidth=3, edgecolor='purple',
                                        facecolor='none', linestyle='-',
                                        label='Ground truth')
            self.ax.add_patch(gt_rect)
            
            # Model prediction
            px1, py1, px2, py2 = self.prediction['box']
            pred_rect = patches.Rectangle((px1, py1), px2-px1, py2-py1,
                                          linewidth=2, edgecolor='white',
                                          facecolor='none', linestyle='--',
                                          label='AI prediction')
            self.ax.add_patch(pred_rect)
            
            # User click
            if self.user_click:
                self.ax.plot(self.user_click[0], self.user_click[1], 'yo', markersize=8,
                            markeredgecolor='black', markeredgewidth=1, label='Your guess')
            
            self.ax.legend(loc='upper left')
            self.ax.set_title("Human vs AI: Complete comparison", fontsize=14)
            self.ax.axis('off')
            
            # Change button text
            self.btn_next.label.set_text('Restart')
            
        else:
            # Reset to beginning
            self.step = 0
            self.user_click = None
            self.btn_next.label.set_text('Next')
            self.update_display()
            return
        
        self.fig.canvas.draw()
    
    def run(self):
        """Run the interactive demo."""
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.15)
        
        # Add Next button
        ax_btn = plt.axes([0.4, 0.05, 0.2, 0.06])
        self.btn_next = Button(ax_btn, 'Next', color='lightblue', hovercolor='skyblue')
        self.btn_next.on_clicked(self.on_next)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial display
        self.update_display()
        
        # Window title
        self.fig.canvas.manager.set_window_title('Fracture Detection Demo')
        
        plt.show()

# Main

def main():
    print("=" * 50)
    print("  FRACTURE DETECTION DEMO")
    print("=" * 50)
    print("\nInstructions:")
    print("1. Click on the X-ray where you think the fracture is")
    print("2. Press 'Next' to see the AI's prediction")
    print("3. Continue pressing 'Next' to see the full comparison")
    print("\n")
    
    demo = FractureDemo(
        image_path=IMAGE_PATH,
        overlay_path=OVERLAY_PATH,
        gt_yolo=GT_YOLO,
        prediction=PREDICTION
    )
    
    demo.run()


if __name__ == "__main__":
    main()
