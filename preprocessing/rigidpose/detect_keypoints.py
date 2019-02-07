"""
Detects keypoints for objects, and writes them to <<DATA_PATH>>/models/models_info.yml
Saves 3D plots to files in <<DATA_PATH>>/models/keypoint_3dplots
"""

import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks

import shutil
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
# DATA_PATH = '/home/lucas/datasets/pose-data/sixd/bop-unzipped/hinterstoisser' # Path to a BOP-SIXD dataset
# TRAIN_SUBDIR = 'train' # Images in this subdir will be used to collect keypoint statistics
DATA_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented' # Path to a BOP-SIXD dataset
TRAIN_SUBDIR = 'train_occl' # Images in this subdir will be used to collect keypoint statistics



models = {}
def get_model(obj):
    if obj not in models:
        models[obj] = inout.load_ply(os.path.join(DATA_PATH, 'models', 'obj_{:02}.ply'.format(obj)))
    return models[obj]

# detector = cv2.FeatureDetector_create("SIFT")
# detector = cv2.xfeatures2d_SIFT()
detector = cv2.ORB_create(nfeatures=20000)

vtx_scores = OrderedDict()
instance_counts = OrderedDict()

# Loop over all sequences
# for seq in ['01']: # ape
# for seq in ['02']: # benchvise
# for seq in ['06']: # cat?
# for seq in ['09']: # duck
# for seq in ['12']: # holepuncher
# for seq in ['13']: # iron
for seq in sorted(os.listdir(os.path.join(DATA_PATH, TRAIN_SUBDIR))):
    info = inout.load_info(os.path.join(DATA_PATH, TRAIN_SUBDIR, seq, 'info.yml'))
    gt = inout.load_gt(os.path.join(DATA_PATH, TRAIN_SUBDIR, seq, 'gt.yml'))
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

        img = cv2.imread(os.path.join(DATA_PATH, TRAIN_SUBDIR, seq, 'rgb', '{:04}.png'.format(frame_idx)))

        # NOTE: Rendering segmentations requires either one of:
        #           Modifying C++ renderer to read BOP annotations
        #           Modify BOP python renderer's shaders to produce seg (hopefully not too hard to reuse shader code from C++ renderer)
        # seg = cv2.imread(os.path.join(DATA_PATH, TRAIN_SUBDIR, seq, 'seg', '{:04}.png'.format(frame_idx)))

        # NOTE: Detector applied on RGB (or is it BGR?) image, i.e. not grayscale. Not sure what the implications of this are.
        all_keypoints = detector.detect(img)

        # print("Total #keypoints: {}".format(len(all_keypoints)))

        # Loop over all object instances
        for instance in gt[frame_idx]:
            if instance['obj_id'] in instance_counts:
                instance_counts[instance['obj_id']] += 1
            else:
                instance_counts[instance['obj_id']] = 1

            model = get_model(instance['obj_id'])
            R_m2c = instance['cam_R_m2c']
            t_m2c = instance['cam_t_m2c']
            bbox = instance['obj_bb'] # xmin, ymin, width, height
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            # print("Object: {}".format(instance['obj_id']))
            # print(xmin, xmax, ymin, ymax)
            # plt.figure()
            # plt.imshow(img)
            # plt.show()

            # Determine which keypoints belong to current object instance
            # OpenCV coordinates correspond to pixel centers according to http://answers.opencv.org/question/35111/origin-pixel-in-the-image-coordinate-system-in-opencv/
            keypoints = []
            for kp in all_keypoints:
                x = int(0.5+kp.pt[0])
                y = int(0.5+kp.pt[1])

                # If seg would be used:
                # if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                #     continue
                # if seg[y,x] != instance['obj_id']:
                #     continue

                # NOTE: Using bbox eliminates need for rendering segmentations. Some keypoints might be outside object, but hopefully these effects are negligible with statistics from enough frames.
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    continue
                keypoints.append(kp)

            if len(keypoints) == 0:
                print("No keypoints found for object {}!".format(instance['obj_id']))
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
    #     break #frame
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

    vtx_scores_filtered[obj_id] = scores
    # vtx_scores_filtered[obj_id] = [scores]

# Run GMM and update models_info.yml
if not os.path.exists(os.path.join(DATA_PATH, 'models', 'models_info_backup.yml')):
    shutil.copyfile(os.path.join(DATA_PATH, 'models', 'models_info.yml'), os.path.join(DATA_PATH, 'models', 'models_info_backup.yml'))
models_info = inout.load_yaml(os.path.join(DATA_PATH, 'models', 'models_info_backup.yml'))

# for obj_id, all_scores in vtx_scores_filtered.items():
#     # scores = np.random.choice(all_scores)
#     # scores = all_scores[0]
#     scores = sum(all_scores) / float(len(all_scores))
for obj_id, scores in vtx_scores_filtered.items():
    print("Inferring GMM for object {}".format(obj_id))

    model = get_model(obj_id)
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

    models_info[obj_id]['kp_x'] = list(map(float, xs))
    models_info[obj_id]['kp_y'] = list(map(float, ys))
    models_info[obj_id]['kp_z'] = list(map(float, zs))

# Update YAML
inout.save_yaml(os.path.join(DATA_PATH, 'models', 'models_info.yml'), models_info)

# TODO: Run on real images. (for instance, bowl object is probably very hard.)
# NOTE: Alternative to LP filtering: #1 aggregate scores from all frames & run. #2 sample some frames and add KP if far from the rest.

os.makedirs(os.path.join(DATA_PATH, 'models', 'keypoint_3dplots'), exist_ok=True)
# for obj_id, all_scores in vtx_scores_filtered.items():
#     # scores = np.random.choice(all_scores)
#     # scores = all_scores[0]
#     scores = sum(all_scores) / float(len(all_scores))
for obj_id, scores in vtx_scores_filtered.items():
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

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot(models_info[obj_id]['kp_x'], models_info[obj_id]['kp_y'], models_info[obj_id]['kp_z'], 'r*', markersize=50)

    plt.savefig(os.path.join(DATA_PATH, 'models', 'keypoint_3dplots', 'obj_{:02}.png'.format(obj_id)))
    # plt.show()