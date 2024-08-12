import torch
from yolov5 import train
from yolov5.utils.general import check_yaml
from yolov5.utils.callbacks import Callbacks
import yaml
import os

# Define your dataset configuration
data_yaml = {
    'train': 'path/to/your/train/images',
    'val': 'path/to/your/val/images',
    'nc': 2,  # number of classes
    'names': ['class1', 'class2']  # class names
}

# Save the data configuration to a YAML file
with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

# Define training parameters
hyp = {
    'lr0': 0.01,
    'lrf': 0.1,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 0.05,
    'cls': 0.5,
    'cls_pw': 1.0,
    'obj': 1.0,
    'obj_pw': 1.0,
    'iou_t': 0.20,
    'anchor_t': 4.0,
    'fl_gamma': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0
}

# Save hyperparameters to a YAML file
with open('hyp.yaml', 'w') as f:
    yaml.dump(hyp, f)

# Main training function
def train_yolov5():
    # Set up training parameters
    opt = {
        'weights': '',  # path to weights file, empty for training from scratch
        'cfg': 'yolov5s.yaml',  # path to model config
        'data': 'data.yaml',  # path to data config
        'hyp': 'hyp.yaml',  # path to hyperparameters file
        'epochs': 100,
        'batch_size': 16,
        'imgsz': 640,
        'rect': False,
        'resume': False,
        'nosave': False,
        'noval': False,
        'noautoanchor': False,
        'evolve': False,
        'bucket': '',
        'cache': False,
        'image_weights': False,
        'device': '',
        'multi_scale': False,
        'single_cls': False,
        'adam': False,
        'sync_bn': False,
        'workers': 8,
        'project': 'runs/train',
        'name': 'exp',
        'exist_ok': False,
        'quad': False,
        'linear_lr': False,
        'label_smoothing': 0.0,
        'upload_dataset': False,
        'bbox_interval': -1,
        'save_period': -1,
        'artifact_alias': 'latest',
        'local_rank': -1,
        'save_dir': 'runs/train/exp',
    }

    # Initialize callbacks
    callbacks = Callbacks()

    # Train the model
    train.run(**opt)  # Remove callbacks from here

if __name__ == '__main__':
    train_yolov5()
