"""Constants for 3DOD."""
import os
import numpy

# Execution
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# Training
IGNORE_IDX_CLS = 1
IGNORE_IDX_REG = 97986 # Nonsense number - expected never to interfere with actual GT annotations

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

# Keypoint instances
NBR_KEYPOINTS = 20

# Classes
DONT_CARE = 'dont_care'

# 3D Boxes
CORNER_COLORS = ['magenta', 'cyan', 'yellow', 'green', 'lime', 'blue', 'purple', 'orange']
#CORNER_NAMES = ("TLF",     "TRF",  "BRF",    "BLF",   "TLR",  "TRR",  "BRR",    "BLR")  # KITTI
BOX_SKELETON = (5, 6, 7, 4, 0, 1, 2, 3, 0, 2, 6, 7, 3, 1, 5, 4)  # With cross on front

KEYPOINT_COLORS = ['magenta', 'cyan', 'yellow', 'green',
                   'lime', 'blue', 'purple', 'orange',
                   'white', 'lightcoral', 'lime', 'olive',
                   'steelblue', 'red', 'gold', 'navy',
                   'dodgerblue', 'mediumaquamarine', 'black', 'gray']
