"""
This script prepares the FracAtlas dataset for YOLO

Requirements:
    shutil
    pathlib
    pandas

"""

import shutil
from pathlib import Path

import pandas as pd

# Configure these paths IF needed 

# Root of the original Figshare dataset (relative to this script)
FRAC_ROOT = Path("FracAtlas")

# Where the images live in the original dataset
IMAGES_ROOT = FRAC_ROOT / "images"          # has "Fractured" and "Non_fractured"
YOLO_ANN_ROOT = FRAC_ROOT / "Annotations" / "YOLO"

# Where the split CSVs live
SPLIT_ROOT = FRAC_ROOT / "Utilities" / "Fracture Split"

# Where it will write the YOLO-style dataset (Ultralytics/YOLOv8/12 format)
OUT_ROOT = Path("datasets") / "fracatlas"

# Column name in train/valid/test CSVs that holds the image filename
# Open one CSV to confirm; change if needed (e.g. "filename", "img_name", etc.)
IMAGE_COL = "image_id"

# The three splits and their CSV files
SPLITS = {
    "train": SPLIT_ROOT / "train.csv",
    "val":   SPLIT_ROOT / "valid.csv",
    "test":  SPLIT_ROOT / "test.csv",
}


def find_image_path(img_name: str) -> Path:
    """
    Find an image in images/Fractured or images/Non_fractured.
    Raises FileNotFoundError if not present.
    """
    for sub in ["Fractured", "Non_fractured"]:
        p = IMAGES_ROOT / sub / img_name
        if p.exists():
            return p
    raise FileNotFoundError(f"Image {img_name} not found in Fractured/ or Non_fractured/")


def prepare_split(split_name: str, csv_path: Path) -> None:
    print(f"\n=== Preparing {split_name} from {csv_path} ===")

    df = pd.read_csv(csv_path)

    if IMAGE_COL not in df.columns:
        raise ValueError(
            f"Column '{IMAGE_COL}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    img_out_dir = OUT_ROOT / "images" / split_name
    lbl_out_dir = OUT_ROOT / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in df[IMAGE_COL]:
        # Ensure the filename has .jpg extension
        img_name = str(img_id)
        if not img_name.lower().endswith(".jpg"):
            img_name = img_name + ".jpg"

        # 1) Copy image into YOLO images/<split>/
        src_img = find_image_path(img_name)
        dst_img = img_out_dir / img_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # 2) Copy or create label into YOLO labels/<split>/
        stem = Path(img_name).stem
        src_lbl = YOLO_ANN_ROOT / f"{stem}.txt"
        dst_lbl = lbl_out_dir / f"{stem}.txt"

        if src_lbl.exists():
            # fractured image → copy YOLO annotation
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # non-fractured image → empty label file (no objects)
            dst_lbl.touch(exist_ok=True)

    print(f"Done {split_name}.")
    print(f"  Images: {img_out_dir}")
    print(f"  Labels: {lbl_out_dir}")


def main():
    for split_name, csv_path in SPLITS.items():
        if not csv_path.exists():
            print(f"Skipping {split_name}: {csv_path} not found")
            continue
        prepare_split(split_name, csv_path)

    print("\nAll splits processed.")
    print("YOLO-ready dataset is at:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
