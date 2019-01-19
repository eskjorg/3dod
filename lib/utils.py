"""Utils"""
import os
import json
from attrdict import AttrDict

import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import Normalize

from lib.constants import SETTINGS_PATH
from lib.constants import TORCHVISION_MEAN, TORCHVISION_STD


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

# Image

def preprocess_image(image_tensor, dims):
    norm = Normalize(mean=TORCHVISION_MEAN, std=TORCHVISION_STD)
    image_tensor = norm(image_tensor / 255)
    image_tensor = image_tensor[:dims[0], :dims[1]]
    return image_tensor.permute(2, 0, 1)

# Load settings

def get_configs(config_name):
    default_config_path = os.path.join(SETTINGS_PATH, 'default_config.json')
    configs = read_json(default_config_path)

    experiment_config_path = os.path.join(SETTINGS_PATH, config_name, 'config.json')
    if os.path.isfile(experiment_config_path):
        configs += read_json(experiment_config_path)
    return configs

def get_layers(config_name):
    path = os.path.join(SETTINGS_PATH, config_name, 'layers.json')
    return read_json(path)

# Geometry

def project_3d_box(size, loc, rot, p_matrix):
    h, _, _ = size
    _, w2, l2 = size / 2
    rot_matrix = np.array([[np.cos(rot), 0, np.sin(rot)],
                           [0, 1, 0],
                           [-np.sin(rot), 0, np.cos(rot)]])
    points_3d = np.ones((4,8))
                                                                #    BLR  BLF BRF BRR  TLR  TLF TRF TRR
    points_3d[:3] = np.tile(loc, (8, 1)).T + rot_matrix @ np.array([[-l2, l2, l2, -l2, -l2, l2, l2, -l2],
                                                                    [0, 0, 0, 0, -h, -h, -h, -h],
                                                                    [w2, w2, -w2, -w2, w2, w2, -w2, -w2]])
    points_2d = p_matrix @ points_3d
    return points_2d[:2] / points_2d[2]
