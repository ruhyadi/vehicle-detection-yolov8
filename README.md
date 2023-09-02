# Vehicle Detection with YOLOv8

## Introduction
YOLOv8 is a real-time object detection model developed by [Ultralytics](https://github.com/ultralytics/ultralytics). This repository demonstrate how to train YOLOv8 on [KITTI](https://www.kaggle.com/datasets/didiruh/capstone-kitti-training) dataset and use it to detect vehicles in images and videos.

## Installation
### Create a virtual environment
We assume that you have [Anaconda](https://www.anaconda.com/) installed. To install the required packages, run the following commands:
```
conda create -n yolov8 python=3.10 cudatoolkit=11.8
conda activate yolov8
```

### Install PyTorch
Next, install PyTorch with the following command:
```
# CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu
```