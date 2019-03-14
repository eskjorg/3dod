"""Utils"""
import os
import json
from attrdict import AttrDict
from importlib import import_module

import numpy as np
import cv2 as cv
import torch
from torchvision.transforms.functional import normalize, to_tensor

from lib.constants import SETTINGS_PATH
from lib.constants import TV_MEAN, TV_STD


## GPU ##

def show_gpu_info():
    """Show GPU info."""
    pass

def get_device():
    """Get best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


## File ops ##

def listdir_nohidden(path):
    fnames = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            fnames.append(f)
    return fnames

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
    image = normalize(to_tensor(image), TV_MEAN, TV_STD)
    if len(image.shape) == 2:
        image._unsqueeze(0)
    return image.flip(0)


def read_velodyne_to_pt(path):
    """Read lidar data from path to pt tensor."""
    pointcloud = np.fromfile(path, dtype=np.float32)
    pointcloud = np.reshape(pointcloud, [-1, 4])
    return torch.from_numpy(pointcloud)


# Modules

def get_class_map(configs):
    return import_module('lib.data.datasets.' + configs.data.dataformat).ClassMap(configs)

def get_metadata(configs):
    return import_module('lib.data.datasets.' + configs.data.dataformat).get_metadata(configs)


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
    layer_spec = read_json(path)
    for layer_name in layer_spec:
        if 'loss_weight' not in layer_spec[layer_name]:
            # Default value: 1.0
            layer_spec[layer_name]['loss_weight'] = 1.0
        if 'cls_specific_heads' not in layer_spec[layer_name]:
            # Default value: False
            layer_spec[layer_name]['cls_specific_heads'] = False
    if 'cls' in layer_spec:
        # For cls head, cls_specific_heads must be False
        assert not layer_spec['cls']['cls_specific_heads']
    return layer_spec

# Geometry

def matrix_from_yaw(yaw):
    return np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])

def project_3d_pts(pts_3d_objframe, p_matrix, loc, rot_y=None, rot_matrix=None):
    rot_matrix = matrix_from_yaw(rot_y) if rot_y else rot_matrix
    nbr_pts = pts_3d_objframe.shape[1]

    # Euclidean transformation to global frame. Store as homogeneous coordinates.
    pts_3d_global = np.ones((4,nbr_pts))
    pts_3d_global[:3] = np.tile(loc, (nbr_pts, 1)).T + rot_matrix @ np.array(pts_3d_objframe)

    # Projection
    pts_2d = p_matrix @ pts_3d_global
    return pts_2d[:2] / pts_2d[2]

def construct_3d_box(size_hwl):
    """Return 3D points of bounding box corners, given parameters."""
    h2, w2, l2 = 0.5 * size_hwl
    # Using nuScenes ordering  # TLF  TRF  BRF  BLF  TLR  TRR  BRR  BLR
    pts_3d_objframe = np.array([[ l2,  l2,  l2,  l2, -l2, -l2, -l2, -l2],
                                [-h2, -h2,  h2,  h2, -h2, -h2,  h2,  h2],
                                [ w2, -w2, -w2,  w2,  w2, -w2, -w2,  w2]])

    return pts_3d_objframe
