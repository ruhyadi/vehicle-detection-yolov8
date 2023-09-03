"""
Convert KITTI format to YOLO format.
usage:
python tools/kitti2yolo.py \
    --images_dir data/kitti/image_2 \
    --labels_dir data/kitti/label_2 \
    --output_dir data/kitti/
"""

import argparse
import os
from glob import glob
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def generate_yolo_labels(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    classes: list = ["Car", "Pedestrian", "Cyclist"],
):
    """Generate YOLO labels."""
    print("Generating YOLO labels...")
    output_dir = os.path.join(output_dir, "labels")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = glob(os.path.join(images_dir, "*.png"))

    for image in tqdm(images, desc="Generating YOLO labels...", total=len(images)):
        frame = Image.open(image)
        image_name = Path(image).name
        image_id = image_name.split(".")[0]
        with open(os.path.join(labels_dir, f"{image_id}.txt"), "r") as f:
            labels = f.readlines()
        with open(os.path.join(output_dir, f"{image_id}.txt"), "w") as f:
            for label in labels:
                label = label.split(" ")
                _class = label[0]
                if _class not in classes:
                    continue
                class_id = classes.index(_class)
                xmin = float(label[4])
                ymin = float(label[5])
                xmax = float(label[6])
                ymax = float(label[7])
                x_center = (((xmax - xmin) / 2) + xmin) / frame.width
                y_center = (((ymax - ymin) / 2) + ymin) / frame.height
                w = (xmax - xmin) / frame.width
                h = (ymax - ymin) / frame.height
                if x_center > 1:
                    x_center = 1.0
                if y_center > 1:
                    y_center = 1.0
                if w > 1:
                    w = 1.0
                if h > 1:
                    h = 1.0
                f.write(f"{class_id} {x_center:.3f} {y_center:.3f} {w:.3f} {h:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Path to images directory."
    )
    parser.add_argument(
        "--labels_dir", type=str, required=True, help="Path to labels directory."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory."
    )
    args = parser.parse_args()
    generate_yolo_labels(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
    )
