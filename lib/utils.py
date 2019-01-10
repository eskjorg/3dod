"""Utils"""
import os
import json
from attrdict import AttrDict
import torch

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

def get_configs(config_name):
    default_config_path = os.path.join(SETTINGS_PATH, 'config.json')
    configs = read_json(default_config_path)

    experiment_config_path = os.path.join(SETTINGS_PATH, config_name, 'config.json')
    configs.update(read_json(experiment_config_path))

    return configs

def get_layers(config_name):
    path = os.path.join(SETTINGS_PATH, config_name, 'layers.json')
    return read_json(path)
