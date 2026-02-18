""" 
This code trains a YOLO12n model with fine-tuned parameters.

Requirements:
    pathlib
    ultralytics
    json
"""


from pathlib import Path
import json

from ultralytics import YOLO

## Tuned Params for learning rate, lr0=0.0015,  lrf=0.01, nano model ##


def main():
    # config 
    data_yaml = "fracatlas.yaml"
    base_model = "yolo12n.pt"

    epochs = 500              # more training
    batch_size = 16           # try 16; if overloads local GPU, switch to 8
    img_size = 640
    device = 0                # GPU

    project = "runs_frac"
    run_name = "yolo12n_fracatlas_tuned"   # new run name

    # 1. load pretrained model 
    model = YOLO(base_model)

    # 2. train / fine-tune on FracAtlas (train/val) 
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        pretrained=True,
        project=project,
        name=run_name,

        # LR TUNING
        lr0=0.0015,    # initial LR (a bit conservative for long training)
        lrf=0.01,     # final LR = lr0 * lrf = 1.5e-5

        # optional: a bit of patience for early stopping (defaults to 50)
        patience=50,
    )

    # directory Ultralytics actually used, e.g. runs_frac/yolo12n_fracatlas_tuned
    save_dir = Path(train_results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    # 3. evaluate best model on the TEST split 
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=data_yaml,
        split="test",
        device=device,
    )

    box = metrics.box
    precision_per_class = box.p.tolist()
    recall_per_class = box.r.tolist()
    mean_precision = float(sum(box.p) / len(box.p)) if len(box.p) else None
    mean_recall = float(sum(box.r) / len(box.r)) if len(box.r) else None
    mAP50 = float(box.map50)

    print("\n=== YOLO12n (tuned) on FracAtlas (test set) ===")
    print(f"Per-class precision: {precision_per_class}")
    print(f"Per-class recall:    {recall_per_class}")
    print(f"Mean precision:      {mean_precision:.4f}")
    print(f"Mean recall:         {mean_recall:.4f}")
    print(f"mAP@0.5:             {mAP50:.4f}")

    # 4. save metrics to JSON next to this run 
    results_dir = save_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "metrics_fracatlas_yolo12n_tuned.json"

    metrics_dict = {
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mAP@0.5": mAP50,
        "class_names": best_model.names,
    }

    with out_path.open("w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nSaved tuned metrics to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
