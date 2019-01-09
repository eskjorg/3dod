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
CONFIG_PATH = os.path.join(SETTINGS_PATH, 'config.json')
