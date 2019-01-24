"""Constants for 3DOD."""
import os
import math
import numpy

# Execution
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# Training
IGNORE_IDX_CLS = 1
IGNORE_IDX_REG = 0

# Math
LN_SQRT_2_PI = math.log(math.sqrt(2*math.pi))
LN_2 = math.log(2)

# Data
ANNOTATION = 'annotation'
INPUT = 'input'
GT_MAP = 'gt_map'
CALIBRATION = 'calibration'
ID = 'id'

# Paths
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
SETTINGS_PATH = os.path.join(PROJECT_PATH, 'settings')

# TorchVision
TV_MEAN = numpy.array((0.485, 0.456, 0.406))
TV_STD = numpy.array((0.229, 0.224, 0.225))

# Matplotlib
PYPLOT_DPI = 100

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
BOX_SKELETON = (3, 7, 4, 0, 1, 2, 6, 5, 1, 6, 7, 4, 5, 2, 3, 0)
