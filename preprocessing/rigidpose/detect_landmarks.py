import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks
from lib.rigidpose.sixd_toolkit.pysixd import inout
from collections import OrderedDict
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from pomegranate import GeneralMixtureModel, MultivariateGaussianDistribution
from mpl_toolkits.mplot3d import Axes3D

# Parameters
DIFFERENTIATE_ON_KP_RESPONSE = False
MAX_NBR_KEYPOINTS = 100
# DIST_TH = 1e-2 # meters
FEATURE_SCALE_FACTOR = 1e-1
MIN_SCORE_SCATTERPLOT = -1
NBR_FRAMES_SAMPLED_PER_SEQ = 100
NBR_GMM_COMPONENTS = 20
SCATTER_VMIN = 0.0
SCATTER_VMAX = 10.0
SCORE_EXP = 1.0
LP_SIGMA_MM = 40.0
LP_DISTMAT_SUBSET_SIZE = 1000
DEPTH_DIFF_TH = 1e-2 # meters
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

vtx_scores = OrderedDict()
instance_counts = OrderedDict()

# Loop over all sequences
# for seq in ['01']: # cat
# for seq in ['02']: # benchvise
# for seq in ['09']: # duck
# for seq in ['12']: # holepuncher
# for seq in ['13']: # iron
for seq in sorted(os.listdir(DATA_PATH)):
    info = inout.load_info(os.path.join(DATA_PATH, seq, 'info.yml'))
    gt = inout.load_gt(os.path.join(DATA_PATH, seq, 'gt.yml'))
    assert len(info) == len(gt)
    nbr_frames = len(info)

    # Loop over all images
    if NBR_FRAMES_SAMPLED_PER_SEQ is None:
        frames = list(range(nbr_frames))
    else:
        frames = sorted(np.random.choice(nbr_frames, NBR_FRAMES_SAMPLED_PER_SEQ))
    for frame_idx in frames:
        print("Frame: {}, objects: {}".format(frame_idx, list(map(lambda x: x['obj_id'], gt[frame_idx]))))
        K = info[frame_idx]['cam_K']
        # Unnecessary:
        # R_w2c = info['cam_R_w2c'] if 'cam_R_w2c' in info else np.eye(3)
        # t_w2c = info['cam_t_w2c'] if 'cam_t_w2c' in info else np.zeros((3,1))
        # info['depth_scale'] also unnecessary, no need to read/scale depth images

        # Loop over all object instances
        for instance in gt[frame_idx]:
            if instance['obj_id'] in instance_counts:
                instance_counts[instance['obj_id']] += 1
            else:
                instance_counts[instance['obj_id']] = 1

            model = get_model(instance['obj_id'])
            R_m2c = instance['cam_R_m2c']
            t_m2c = instance['cam_t_m2c']

            # TODO: Run feature detection only once per frame!
            img = cv2.imread(os.path.join(DATA_PATH, seq, 'rgb', '{:04}.png'.format(frame_idx)))
            keypoints = detector.detect(img)
            if len(keypoints) == 0:
                print("No keypoints found!")
                continue

            sigma = np.array(list(map(lambda x: x.size, keypoints)))
            # Fairly confident order is correct - 1st row is horizontal coordinate, 2nd row is vertical
            x_2d = np.array(list(map(lambda x: x.pt, keypoints))).T
            detection_score = np.array(list(map(lambda x: x.response, keypoints)))

            # Find top-k keypoints based on response
            score_sorted = np.sort(detection_score)
            strong_kp_mask = detection_score >= score_sorted[max(0, len(score_sorted)-MAX_NBR_KEYPOINTS)]

            # Select only these top-k keypoints
            sigma = sigma[strong_kp_mask]
            sigma *= FEATURE_SCALE_FACTOR
            x_2d = x_2d[:, strong_kp_mask]
            detection_score = detection_score[strong_kp_mask]

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

            # Filter vertices based on depth
            # Shape: (nbr_kp, nbr_vtx)
            small_depth_vtx_mask = parallel_coordinates >= np.min(parallel_coordinates, axis=1, keepdims=True) + DEPTH_DIFF_TH

            # Non-vectorized implementation.
            # if instance['obj_id'] not in vtx_scores:
            #     vtx_scores[instance['obj_id']] = np.zeros((nbr_vtx,))
            # for kp_idx in range(nbr_kp):
            #     print(kp_idx)
            #     for vtx_idx in range(nbr_vtx):
            #         if small_depth_vtx_mask[kp_idx, vtx_idx]:
            #             dist_to_ray = np.linalg.norm(all_vtx[:,vtx_idx] - parallel_coordinates[kp_idx, vtx_idx]*vrays[:,kp_idx])
            #             # TODO: Maybe abort iteration if distance big?
            #             dist_weight = np.exp(-dist_to_ray**2 / (2.0*(0.5*(K[0,0]+K[1,1])*sigma[kp_idx])**2))
            #             vtx_scores[instance['obj_id']][vtx_idx] += detection_score[kp_idx] * dist_weight

            # Shape: (3, nbr_kp, nbr_vtx)
            # all_vtx_parallel = parallel_coordinates[np.newaxis,:,:] * vrays[:,:,np.newaxis]
            # all_vtx_orthogonal = all_vtx[:,np.newaxis,:] - all_vtx_parallel
            # 
            # # Shape: (nbr_kp, nbr_vtx)
            # dists = np.linalg.norm(all_vtx_orthogonal, axis=0)
            
            all_vtx_cam_frame = R_m2c @ all_vtx + t_m2c
            all_vtx_proj = K @ all_vtx_cam_frame
            all_vtx_proj_pixels = all_vtx_proj[0:2,:] / all_vtx_proj[np.newaxis,2,:]
            dists = np.linalg.norm(all_vtx_proj_pixels[:,np.newaxis,:] - x_2d[:,:,np.newaxis], axis=0)

            # Shape: (nbr_kp, nbr_vtx)
            dist_weight = np.exp(-dists**2 / (2.0*(sigma[:,np.newaxis])**2))

            # Shape: (nbr_vtx,)
            if DIFFERENTIATE_ON_KP_RESPONSE:
                scores = np.sum(small_depth_vtx_mask * dist_weight * detection_score[:,np.newaxis], axis=0)
            else:
                scores = np.sum(small_depth_vtx_mask * dist_weight, axis=0)

            if instance['obj_id'] in vtx_scores:
                vtx_scores[instance['obj_id']].append(scores)
            else:
                vtx_scores[instance['obj_id']] = [scores]

            # if instance['obj_id'] in vtx_scores:
            #     k = instance_counts[instance['obj_id']]
            #     old_scores = vtx_scores[instance['obj_id']]
            #     vtx_scores[instance['obj_id']] = (k-1.0)/k*old_scores + 1.0/k*scores
            # else:
            #     vtx_scores[instance['obj_id']] = scores

            # break #instance
        # if frame_idx > 10:
        #     break
        # break #frame
    # break #seq

from scipy.spatial.distance import cdist, pdist, squareform
vtx_scores_filtered = OrderedDict()
for obj_id, all_scores in vtx_scores.items():
    scores_raw = sum(all_scores) / float(len(all_scores))
    model = get_model(obj_id)

    nbr_vtx = model['pts'].shape[0]
    vtx_subset = np.random.choice(range(nbr_vtx), LP_DISTMAT_SUBSET_SIZE)
    # distance_matrix = squareform(pdist(model['pts'], metric='euclidean'))
    distance_matrix = cdist(model['pts'][vtx_subset], model['pts'], metric='euclidean')
    kernel = np.exp(-0.5*(distance_matrix / LP_SIGMA_MM)**2)
    scores_lowpass = np.sum(scores_raw[vtx_subset, np.newaxis] * kernel, axis=0) / np.sum(kernel, axis=0)

    # scores = scores_raw - scores_lowpass
    scores = scores_raw / scores_lowpass
    # scores = scores_lowpass

    print(np.min(scores))
    print(np.max(scores))

    vtx_scores_filtered[obj_id] = [scores]

# TODO: Save keypoints to YAML
# TODO: Run on real images. (for instance, bowl object is probably very hard.)
# NOTE: Alternative to LP filtering: #1 aggregate scores from all frames & run. #2 sample some frames and add KP if far from the rest.

# for obj_id, all_scores in vtx_scores.items():
for obj_id, all_scores in vtx_scores_filtered.items():
    # scores = np.random.choice(all_scores)
    # scores = all_scores[0]
    scores = sum(all_scores) / float(len(all_scores))

    print("Plotting object {}".format(obj_id))

    model = get_model(obj_id)

    # plt.figure()
    # plt.hist(scores, bins=100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # xs, ys, zs = model['pts'].T
    xs, ys, zs = model['pts'][scores >= MIN_SCORE_SCATTERPLOT, :].T
    cvals = scores[scores >= MIN_SCORE_SCATTERPLOT]**SCORE_EXP
    # cvals -= np.min(cvals)
    # cvals /= np.max(cvals)
    ax.scatter(
        xs,
        ys,
        zs,
        c=cvals,
        cmap=plt.get_cmap('Greens'),
        norm=colors.Normalize(vmin=SCATTER_VMIN**SCORE_EXP, vmax=SCATTER_VMAX**SCORE_EXP, clip=False),
        # norm=colors.Normalize(),
        # norm=colors.LogNorm(),
    )

    # GMM
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    X = model['pts'][scores >= MIN_SCORE_SCATTERPLOT, :]
    # X = X[np.random.choice(X.shape[0], 1000),:]
    model = GeneralMixtureModel.from_samples(
        MultivariateGaussianDistribution,
        n_components=NBR_GMM_COMPONENTS,
        init='kmeans++',
        # init='random',
        X=X,
        weights=scores[scores >= MIN_SCORE_SCATTERPLOT]**SCORE_EXP,
    )
    xs, ys, zs = np.array([d.mu for d in model.distributions]).T
    ax.plot(xs, ys, zs, 'r*', markersize=50)

    plt.show()
