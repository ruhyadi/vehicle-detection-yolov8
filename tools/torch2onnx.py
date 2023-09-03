"""
Convert YOLOv8 PyTorch to ONNX model.
usage:
python tools/torch2onnx.py \
    --weights_path tmp/vehicle_kitti_v0_last.pt
"""

import argparse

from ultralytics import YOLO


def convert_torch_to_onnx(
    weights_path: str,
    fp16: bool = False,
    dynamic: bool = True,
) -> None:
    """Convert YOLOv8 PyTorch to ONNX model."""
    # load model
    model = YOLO(weights_path)

    # convert model
    model.export(format="onnx", half=fp16, dynamic=dynamic, simplify=True)


if __name__ == "__main__":
    """Convert YOLOv8 PyTorch to ONNX model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True, help="weights path")
    parser.add_argument(
        "--fp16", action="store_true", help="convert model to half precision"
    )
    parser.add_argument("--dynamic", action="store_true", default=True, help="dynamic ONNX axes")
    args = parser.parse_args()

    convert_torch_to_onnx(
        weights_path=args.weights_path,
        fp16=args.fp16,
        dynamic=args.dynamic,
    )
