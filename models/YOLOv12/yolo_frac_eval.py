"""
This code evaluates a trained YOLO12n model on either the train, val, or test set. 
Output is precision, recall, and mAP@0.5
You have to specify which set of data to run it on.
-----------------------------------------------------------------------------------
Requirements:
    pathlib
    json
    ultralytics YOLO

    Weights from training a YOLO12n model

"""


from pathlib import Path
import json

from ultralytics import YOLO

# Eval Script to quickly get the stats with the results of a model #

def main():
    data_yaml = "fracatlas.yaml"
    device = 0  # GPU

    # choose the run you want to evaluate
    save_dir = Path("runs_frac") / "yolo12n_fracatlas_tuned"   # adjust path
    best_weights = save_dir / "weights" / "best.pt"            # adjust path

    # evaluate best model on the data split of your choice split 
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=data_yaml,
        split="test",        # determines which split to run it on
        device=device,
        conf = 0.5           # set's confidence threshold to filter out false positives, this can be tuned by the user
    )

    box = metrics.box
    precision_per_class = box.p.tolist()
    recall_per_class = box.r.tolist()
    mean_precision = float(sum(box.p) / len(box.p)) if len(box.p) else None
    mean_recall = float(sum(box.r) / len(box.r)) if len(box.r) else None
    mAP50 = float(box.map50)

    print("\n=== YOLO12n on FracAtlas (test set) ===")
    print(f"Per-class precision: {precision_per_class}")
    print(f"Per-class recall:    {recall_per_class}")
    print(f"Mean precision:      {mean_precision:.4f}")
    print(f"Mean recall:         {mean_recall:.4f}")
    print(f"mAP@0.5:             {mAP50:.4f}")

    # save metrics next to the run
    #out_path = save_dir / "metrics_fracatlas_yolo12n_test.json"
    """metrics_dict = {
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mAP@0.5": mAP50,
        "class_names": best_model.names,
    #}"""
    #with out_path.open("w") as f:
        #json.dump(metrics_dict, f, indent=2)

    #print(f"\nSaved metrics to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
