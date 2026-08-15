"""
Bone Fracture Detection — Gradio demo
Loads the fine-tuned YOLOv12-nano checkpoint (trained on FracAtlas) and
runs fracture localization on an uploaded X-ray image.
"""

import gradio as gr
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "models/YOLOv12/best.pt"
model = YOLO(MODEL_PATH)

EXAMPLES_DIR = "examples"


def detect_fracture(image: Image.Image, conf: float):
    if image is None:
        return None, "Upload an X-ray image to run detection."

    results = model.predict(image, conf=conf, verbose=False)
    result = results[0]
    annotated = Image.fromarray(result.plot()[:, :, ::-1])

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        summary = "No fracture detected above the selected confidence threshold."
    else:
        lines = [f"**{len(boxes)} fracture region(s) detected:**"]
        for i, b in enumerate(boxes, start=1):
            score = float(b.conf[0])
            x1, y1, x2, y2 = [round(v, 1) for v in b.xyxy[0].tolist()]
            lines.append(f"{i}. confidence {score:.2f} — box [{x1}, {y1}, {x2}, {y2}]")
        summary = "\n".join(lines)

    return annotated, summary


with gr.Blocks(title="Bone Fracture Detection") as demo:
    gr.Markdown(
        """
        # 🦴 Bone Fracture Detection
        Fine-tuned **YOLOv12-nano** object detector trained on the
        [FracAtlas](https://www.nature.com/articles/s41597-023-02432-4) musculoskeletal
        X-ray dataset. Upload an X-ray to localize possible fractures.

        Test-set performance (conf=0.5): **mAP\@0.5 = 0.62**, precision = 0.78, recall = 0.46,
        average IoU = 0.71.

        > For research/demo purposes only — not a medical diagnostic tool.
        """
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="X-ray image")
            conf_slider = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Confidence threshold")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detection result")
            out_text = gr.Markdown()

    btn.click(detect_fracture, inputs=[inp, conf_slider], outputs=[out_img, out_text])
    inp.change(detect_fracture, inputs=[inp, conf_slider], outputs=[out_img, out_text])

if __name__ == "__main__":
    demo.launch()
