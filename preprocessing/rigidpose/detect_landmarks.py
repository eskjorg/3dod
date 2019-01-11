import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks
from lib.rigidpose.sixd_toolkit.pysixd import inout
import numpy as np

# Path to a BOP-SIXD dataset
data_path = '/home/lucas/datasets/pose-data/bop/datasets/hinterstoisser/train'
model_path = '/home/lucas/datasets/pose-data/bop/datasets/hinterstoisser/models'

models = {}
def get_model(obj):
    if obj not in models:
        models[obj] = inout.load_ply(os.path.join(model_path, 'obj_{:02}.ply'.format(obj)))
    return models[obj]

# Loop over all sequences
for seq in os.listdir(data_path):
    info = inout.load_info(os.path.join(data_path, seq, 'info.yml'))
    gt = inout.load_gt(os.path.join(data_path, seq, 'gt.yml'))
    assert len(info) == len(gt)
    nbr_frames = len(info)

    # Loop over all images
    for frame_idx in list(range(nbr_frames)):
        # Loop over all object instances
        for instance in gt[frame_idx]:
            model = get_model(instance['obj_id'])
            K = instance['cam_K'] if 'cam_K' in instance else info[frame_idx]['cam_K']
            R_m2c = instance['cam_R_m2c']
            t_m2c = instance['cam_t_m2c']
            R_w2c = info['cam_R_w2c'] if 'cam_R_w2c' in info else np.eye(3)
            t_w2c = info['cam_t_w2c'] if 'cam_t_w2c' in info else np.zeros((3,1))

            # Apply SIFT detector
            # Normalize pixel coordinates using calibration
            # Determine viewing ray from 2D point, transform from camera coordinate system to object coordinate system using inverse annotated pose
            # Find all vertices within some distance from line (radius depends on SIFT scale)
            # Filter vertices based on depth along viewing ray (project to viewing ray, obtaining the coordinate lambda along this axis). Select only vertices for which lambda is close enough to lambda_min (mm threshold).

            # Each vertex gets a vote depending on:
                # Detector score
                # Distance from line - gaussian kernel weighting according to SIFT scale?

        # Pick top-n vertices for object
