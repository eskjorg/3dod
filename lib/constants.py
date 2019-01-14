"""Constants for 3DOD."""
import os
import math

# Execution
TRAIN = 'train'
VAL = 'val'

# Training
IGNORE_IDX_CLS = 1
IGNORE_IDX_REG = 0

# Math
LN_SQRT_2_PI = math.log(math.sqrt(2*math.pi))

# Data
ANNOTATION = 'annotation'
INPUT = 'input'
GT_MAP = 'gt_map'
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

# Classes
DONT_CARE = 'dont_care'

# Geometry
KEYPOINT_NAME_MAP = ("BLR", "BLF", "BRF", "BRR", "TLR", "TLF", "TRF", "TRR")
