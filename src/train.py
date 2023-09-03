"""
Train YOLOv8 on KITTI.
usage:
python src/train.py \
    --weights yolov8s.pt \
    --config configs/dataset/kitti.yaml \
    --epochs 15 \
    --batch-size 4 \
    --img-size 640 \
    --device 0 \
    --workers 4
"""

import rootutils

ROOT = rootutils.autosetup()

import argparse
from typing import Union

from ultralytics import YOLO


def train(
    weights: str = "yolov8s.pt",
    config: str = "configs/dataset/kitti.yaml",
    epochs: int = 15,
    batch_size: int = 8,
    img_size: int = 640,
    device: Union[str, int] = 0,
    workers: int = 4,
    resume: bool = False,
) -> None:
    """Train YOLOv8 model."""

    # Load a model
    model = YOLO(weights)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
    )


if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov8s.pt", help="weights path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/kitti.yaml",
        help="data config file path",
    )
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="size of each image dimension"
    )
    parser.add_argument(
        "--device", default=0, help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument("--resume", action="store_true", help="resume training flag")
    opt = parser.parse_args()

    train(
        weights=opt.weights,
        config=opt.config,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        device=opt.device,
        workers=opt.workers,
        resume=opt.resume,
    )
