"""Utils"""
import os
import json
from attrdict import AttrDict

import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import ToTensor

from lib.constants import SETTINGS_PATH


## GPU ##

def show_gpu_info():
    """Show GPU info."""
    pass

def get_device():
    """Get best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


## File ops ##

def read_json(path):
    """Read json file to AttrDict."""
    with open(path) as file:
        json_dict = json.loads(file.read())
    return AttrDict(json_dict)


def read_image_to_pt(path, load_type=cv.IMREAD_COLOR):
    """Read an image from path to pt tensor."""
    image = cv.imread(path, load_type)
    if image is None:
        raise Exception('Failed to read image: {}.'.format(path))
    image = image.astype(np.float32)
    if len(image.shape) == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    return torch.from_numpy(image)

def read_velodyne_to_pt(path):
    """Read lidar data from path to pt tensor."""
    pointcloud = np.fromfile(path, dtype=np.float32)
    pointcloud = np.reshape(pointcloud, [-1, 4])
    return torch.from_numpy(pointcloud)

# Load settings

def get_configs(config_name):
    default_config_path = os.path.join(SETTINGS_PATH, 'default_config.json')
    configs = read_json(default_config_path)

    experiment_config_path = os.path.join(SETTINGS_PATH, config_name, 'config.json')
    if os.path.isfile(experiment_config_path):
        configs.update(read_json(experiment_config_path))

    return configs

def get_layers(config_name):
    path = os.path.join(SETTINGS_PATH, config_name, 'layers.json')
    return read_json(path)
