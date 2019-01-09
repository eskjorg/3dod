"""Parsing input arguments."""
import argparse
import logging
import os
import json
import shutil
import torch

from lib.constants import PROJECT_PATH
from lib.utils import show_gpu_info


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='3D object detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-load-path', default='3dod_demo',
                        help='name of the config dir that is going to be used')
    parser.add_argument('--checkpoint-load-path', default='',
                        help='path of the model to load')
    parser.add_argument('--name', default='3dod_demo',
                        help='name of the execution, will be '
                             'the name of the experiment\'s directory')
    parser.add_argument('--root-path', default=get_default_root(),
                        help='the root directory to hold experiments')

    args = parser.parse_args()

    experiment_path = os.path.join(args.root_path, args.name)
    args.experiment_path = experiment_path
    args.checkpoint_root_dir = os.path.join(experiment_path, 'checkpoints')
    os.makedirs(args.checkpoint_root_dir, exist_ok=True)

    return args


def get_default_root():
    """Get default root."""
    project_root_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(project_root_path, 'experiments')


def setup_logging(experiment_path, mode):
    """Setup logging."""
    logs_path = os.path.join(experiment_path, 'logs')
    log_file_name = '{}.log'.format(mode)
    os.makedirs(logs_path, exist_ok=True)
    log_path = os.path.join(logs_path, log_file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(fmt='%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.info('Log file is %s', log_path)


def prepare_environment():
    """Prepare environment."""
    os.environ['TORCH_HOME'] = os.path.join(PROJECT_PATH, '.torch')
    cuda_is_available = torch.cuda.is_available()
    logging.info('Use cuda: %s', cuda_is_available)
    if cuda_is_available:
        show_gpu_info()
        torch.backends.cudnn.benchmark = True


def save_settings(args):
    """Save user settings to experiment's setting directory."""
    experiment_settings_path = os.path.join(args.experiment_path, 'settings')
    logging.info('Save settings to %s', experiment_settings_path)

    shutil.rmtree(experiment_settings_path, ignore_errors=True)
    shutil.copytree(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings'),
                    experiment_settings_path)

    with open(os.path.join(experiment_settings_path, 'args.json'), 'w') as file:
        json.dump(vars(args), file, indent=4)
