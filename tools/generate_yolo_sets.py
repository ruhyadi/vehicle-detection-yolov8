"""
Generate YOLO train val sets.
Usage: 
python tools/generate_yolo_sets.py \
    --images_dir data/kitti/images \
    --output_dir data/kitti \
    --train_val_split 0.80 \
    --prefix yolo
"""

import argparse
import os
from glob import glob
from pathlib import Path


def generate_yolo_sets(
    images_dir: str,
    output_dir: str,
    train_val_split: float = 0.85,
    prefix: str = "yolo",
):
    """Generate YOLO train val sets."""
    images = glob(os.path.join(images_dir, "*.png"))
    train_val_split = int(len(images) * train_val_split)
    train_images = images[:train_val_split]
    val_images = images[train_val_split:]
    with open(os.path.join(output_dir, f"{prefix}_train.txt"), "w") as f:
        for image in train_images:
            f.write(f"./images/{Path(image).name}\n")
    with open(os.path.join(output_dir, f"{prefix}_val.txt"), "w") as f:
        for image in val_images:
            f.write(f"./images/{Path(image).name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Path to images directory."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory."
    )
    parser.add_argument(
        "--train_val_split", type=float, default=0.85, help="Train val split."
    )
    parser.add_argument(
        "--prefix", type=str, default="yolo", help="Prefix for train and val files."
    )
    args = parser.parse_args()
    generate_yolo_sets(
        args.images_dir,
        args.output_dir,
        args.train_val_split,
    )
