import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks
from lib.rigidpose.sixd_toolkit.pysixd import inout
import numpy as np
import cv2

# Parameters
DIST_TH = 1e-2 # meters
DATA_PATH = '/home/lucas/datasets/pose-data/bop/datasets/hinterstoisser/train' # Path to a BOP-SIXD dataset
MODEL_PATH = '/home/lucas/datasets/pose-data/bop/datasets/hinterstoisser/models'



models = {}
def get_model(obj):
    if obj not in models:
        models[obj] = inout.load_ply(os.path.join(MODEL_PATH, 'obj_{:02}.ply'.format(obj)))
    return models[obj]

# detector = cv2.FeatureDetector_create("SIFT")
# detector = cv2.xfeatures2d_SIFT()
detector = cv2.ORB_create()

vtx_scores = {}

# Loop over all sequences
for seq in os.listdir(DATA_PATH):
    info = inout.load_info(os.path.join(DATA_PATH, seq, 'info.yml'))
    gt = inout.load_gt(os.path.join(DATA_PATH, seq, 'gt.yml'))
    assert len(info) == len(gt)
    nbr_frames = len(info)

    # Loop over all images
    for frame_idx in list(range(nbr_frames)):
        K = info[frame_idx]['cam_K']
        # Unnecessary:
        # R_w2c = info['cam_R_w2c'] if 'cam_R_w2c' in info else np.eye(3)
        # t_w2c = info['cam_t_w2c'] if 'cam_t_w2c' in info else np.zeros((3,1))
        # info['depth_scale'] also unnecessary, no need to read/scale depth images

        # Loop over all object instances
        for instance in gt[frame_idx]:
            model = get_model(instance['obj_id'])
            R_m2c = instance['cam_R_m2c']
            t_m2c = instance['cam_t_m2c']

            img = cv2.imread(os.path.join(DATA_PATH, seq, 'rgb', '{:04}.png'.format(frame_idx)))
            keypoints = detector.detect(img)

            # TODO: Filter keypoints - use only top-k based on detection score
            # TODO: Filter keypoints - use only top-k based on detection score
            # TODO: Filter keypoints - use only top-k based on detection score
            # TODO: Filter keypoints - use only top-k based on detection score
            # TODO: Filter keypoints - use only top-k based on detection score

            sigma = np.array(list(map(lambda x: x.size, keypoints)))
            # Fairly confident order is correct - 1st row is horizontal coordinate, 2nd row is vertical
            x_2d = np.array(list(map(lambda x: x.pt, keypoints))).T

            nbr_kp = x_2d.shape[1]

            # Homogeneous coordinates
            x = np.concatenate(
                (x_2d, np.ones((1, nbr_kp))),
                axis=0,
            )

            # Normalize pixel coordinates using calibration
            # 3D points in image plane, camera coordinate system
            X_imgplane_cam = np.linalg.solve(K, x)
            # Transform to model coordinate system & normalize to get viewing rays.
            X_imgplane = R_m2c.T @ (X_imgplane_cam - t_m2c)
            # Shape: (3, nbr_kp)
            vrays = X_imgplane / np.linalg.norm(X_imgplane, axis=0)

            # Project vertices to 2 components, one parallel and one orthogonal to viewing ray.
            # Shape: (3, nbr_vtx)
            all_vtx = model['pts'].T
            nbr_vtx = all_vtx.shape[1]

            # Dot product along axis 0 of arrays with shape: (3, nbr_kp, nbr_vtx)
            # Shape: (nbr_kp, nbr_vtx)
            parallel_coordinates = np.sum(vrays[:,:,np.newaxis]*all_vtx[:,np.newaxis,:], axis=0)

            # Shape: (3, nbr_kp, nbr_vtx)
            all_vtx_parallel = parallel_coordinates[np.newaxis,:,:] * vrays[:,:,np.newaxis]
            all_vtx_orthogonal = all_vtx[:,np.newaxis,:] - all_vtx_parallel

            # Shape: (nbr_kp, nbr_vtx)
            dists = np.linalg.norm(all_vtx_orthogonal, axis=0)

            # TODO: Find all vertices within some distance from line (radius depends on SIFT scale)
            # TODO: Find all vertices within some distance from line (radius depends on SIFT scale)
            # TODO: Find all vertices within some distance from line (radius depends on SIFT scale)
            # TODO: Find all vertices within some distance from line (radius depends on SIFT scale)
            # TODO: Find all vertices within some distance from line (radius depends on SIFT scale)

            # TODO: Filter vertices based on depth along viewing ray (project to viewing ray, obtaining the coordinate lambda along this axis). Select only vertices for which lambda is close enough to lambda_min (mm threshold).
            # TODO: Filter vertices based on depth along viewing ray (project to viewing ray, obtaining the coordinate lambda along this axis). Select only vertices for which lambda is close enough to lambda_min (mm threshold).
            # TODO: Filter vertices based on depth along viewing ray (project to viewing ray, obtaining the coordinate lambda along this axis). Select only vertices for which lambda is close enough to lambda_min (mm threshold).
            # TODO: Filter vertices based on depth along viewing ray (project to viewing ray, obtaining the coordinate lambda along this axis). Select only vertices for which lambda is close enough to lambda_min (mm threshold).
            # TODO: Filter vertices based on depth along viewing ray (project to viewing ray, obtaining the coordinate lambda along this axis). Select only vertices for which lambda is close enough to lambda_min (mm threshold).

            dist_weight = np.sum(np.exp(-dists**2 / (2.0*(0.5*(K[0,0]+K[1,1])*sigma[:,np.newaxis])**2)), axis=0)
            if instance['obj_id'] in vtx_scores:
                vtx_scores[instance['obj_id']]['detection_score'] += None
                vtx_scores[instance['obj_id']]['dist_weight'] += dist_weight
            else:
                vtx_scores[instance['obj_id']] = {
                    'detection_score': detection_score,
                    'dist_weight': dist_weight,
                }
            # for ray_idx in range(nbr_kp):
            #     vtx_close = []
            #     for vtx_idx in range(nbr_vtx):
            #         if dists[ray_idx, vtx_idx] < DIST_TH:
            #             vtx_scores[obj][vtx_idx] += dists[ray_idx, vtx_idx]

            break #instance
        break #frame
    break #seq

print(vtx_scores[6].shape)
# TODO: Pick top-n vertices for instance
# TODO: Pick top-n vertices for instance
# TODO: Pick top-n vertices for instance
# TODO: Pick top-n vertices for instance
# TODO: Pick top-n vertices for instance
