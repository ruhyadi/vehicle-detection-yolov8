# Vehicle Detection with YOLOv8

## Introduction
YOLOv8 is a real-time object detection model developed by [Ultralytics](https://github.com/ultralytics/ultralytics). This repository demonstrate how to train YOLOv8 on [KITTI](https://www.kaggle.com/datasets/didiruh/capstone-kitti-training) dataset and use it to detect vehicles in images and videos.

## Installation
### Create a virtual environment
We assume that you have [Anaconda](https://www.anaconda.com/) installed. To install the required packages, run the following commands:
```bash
conda create -n yolov8 python=3.10 cudatoolkit=11.8
conda activate yolov8
```

### Install PyTorch
Next, install PyTorch with the following command:
```bash
# CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Download KITTI dataset
Download the [KITTI](https://www.kaggle.com/datasets/didiruh/capstone-kitti-training) dataset and extract it to the `data` folder.

You should have the following folder structure:
```bash
data
├── kitti
│   ├── image_2 # images from the left color camera
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   ├── ...
│   │   └── 007480.png
│   ├── label_2 # label files in KITTI format
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   ├── ...
│   │   └── 007480.txt
│   ├── calib # calibration files
│   │   ├── 000000.txt
|   │   ├── 000001.txt
│   │   ├── ...
│   │   └── 007480.txt
```

In order to train YOLOv8 with KITTI dataset, the first step we need to rename `image_2` to `images`. You can do this by running the following command:
```bash
mv data/kitti/image_2 data/kitti/images
```
You should have the following folder structure:
```bash
data
├── kitti
│   ├── images # images from the left color camera
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   ├── ...
│   │   └── 007480.png
│   ├── label_2 # label files in KITTI format
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   ├── ...
│   │   └── 007480.txt
│   ├── calib # calibration files
│   │   ├── 000000.txt
|   │   ├── 000001.txt
│   │   ├── ...
│   │   └── 007480.txt
```

### Convert KITTI format to YOLO format
KITTI dataset uses a different format than YOLO. To convert the KITTI format to YOLO format, run the following command:
```bash
python tools/kitti2yolo.py \
    --images_dir data/kitti/images \
    --labels_dir data/kitti/label_2 \
    --output_dir data/kitti
```
You should have the following folder structure:
```bash
data
├── kitti
│   ├── images
|   ├── label_2
│   ├── calib
│   └── labels # label files in YOLO format
│       ├── 000000.txt
│       ├── 000001.txt
│       ├── ...
│       └── 007480.txt
```

### Create YOLO Training and Validation sets
To create the YOLO training and validation sets, run the following command:
```bash
python scripts/generate_yolo_sets.py \
    --images_dir data/kitti/images \
    --output_dir data/kitti \
    --train_val_split 0.80 \
    --prefix yolo
```
You should have the following folder structure:
```bash
data
├── kitti
│   ├── images
|   ├── label_2
│   ├── calib
│   ├── labels
│   ├── yolo_train.txt # YOLO training set
│   └── yolo_val.txt # YOLO validation set
```

### Create a YAML Configuration File
Create a YAML configuration file for training. You can use the `dataset/yolov8s.yml` file as a template. The configuration file should be placed in the `configs` folder.

## Training
All sets are ready. Now, we can start training. To train YOLOv8, run the following command:
```bash
python src/train.py \
    --weights yolov8s.pt \
    --config configs/dataset/kitti.yaml \
    --epochs 15 \
    --batch-size 4 \
    --img-size 640 \
    --device 0 \
    --workers 4
```
If you want to resume training from a checkpoint, add the `--resume` flag:
```bash
python src/train.py \
    --weights yolov8s.pt \
    --config configs/dataset/kitti.yaml \
    --epochs 15 \
    --batch-size 4 \
    --img-size 640 \
    --device 0 \
    --workers 4 \
    --resume
```
You can also use the `--weights` flag to specify a custom weight file.

## Export to ONNX
ONNX (Open Neural Network Exchange) is an open format for representing deep learning models. To export the trained model to ONNX, run the following command:
```bash
python tools/torch2onnx.py \
    --weights_path tmp/vehicle_kitti_v0_last.pt
```
You should have exported ONNX model in the `tmp` folder.