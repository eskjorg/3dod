"""Constants for 3DOD."""
import os

# Execution
TRAIN = 'train'
VAL = 'val'

# Data
ANNOTATION = 'annotation'
INPUT = 'input'
MASK = 'mask'
CALIBRATION = 'calibration'
ID = 'id'

# Paths
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
SETTINGS_PATH = os.path.join(PROJECT_PATH, 'settings')

# KITTI
LEFT_COLOR = 'left_color'
RIGHT_COLOR = 'right_color'
LEFT_GRAY = 'left_gray'
RIGHT_GRAY = 'right_gray'
VELODYNE = 'velodyne'
LABEL_2 = 'label_2'
CALIB = 'calib'
