"""Run thresholding and nms."""
from attrdict import AttrDict

import torch
from maskrcnn_benchmark.layers import nms

import numpy as np
import math

from lib.constants import NBR_KEYPOINTS
from lib.data import maps
from lib.utils import get_device, get_class_map, get_metadata
from lib.postprocessing import RunnerIf
from lib.rigidpose.pose_estimator import ransac, normalize, RANSACException, resec3pts, pflat, pextend
import pyopengv

def invert_eucl(eucl):
    return np.concatenate([eucl[:,:3].T, -eucl[:,:3].T@eucl[:,[3]]], axis=1)

def p3p_kneip(u, U):
    viewing_rays = u / np.linalg.norm(u, axis=0)
    cameras = pyopengv.absolute_pose_p3p_kneip(viewing_rays.T, U.T)
    cameras = [invert_eucl(cam) for cam in cameras if np.all(np.isfinite(cam))]
    return cameras

def opengv_ransac(u, U):
    viewing_rays = u / np.linalg.norm(u, axis=0)
    method_name = 'KNEIP' # This is the default
    # method_name = 'EPNP'
    cam = pyopengv.absolute_pose_ransac(viewing_rays.T, U.T, method_name, 0.02, iterations=1000, probability=0.99)
    cam = invert_eucl(cam)
    return cam if np.all(np.isfinite(cam)) else None

def epnp(u, U):
    viewing_rays = u / np.linalg.norm(u, axis=0)
    cam = pyopengv.absolute_pose_epnp(viewing_rays.T, U.T)
    cam = invert_eucl(cam)
    return cam if np.all(np.isfinite(cam)) else None

class GroupedCorrespondenceSet():
    def __init__(self, K):
        self.K = K
        self.group_ids = []
        self._group_id2group_idx_dict = {}
        self.grouped_idx_lists = []
        self.u = np.empty((3, 0))
        self.U = np.empty((4, 0))
        self.b_per_sample = np.empty((2, 0,))

    @property
    def nbr_samples(self):
        return self.u.shape[1]

    @property
    def nbr_groups(self):
        return len(self.grouped_idx_lists)

    @property
    def sample_confidences(self):
        return self._packed['sample_confidences']

    @property
    def u_grouped(self):
        return self._packed['u_grouped']

    @property
    def U_grouped(self):
        return self._packed['U_grouped']

    @property
    def sample_confidences_grouped(self):
        return self._packed['sample_confidences_grouped']

    @property
    def group_confidences(self):
        return self._packed['group_confidences']

    def group_id2group_idx(self, group_id):
        return self._group_id2group_idx_dict[group_id]

    def nbr_samples_in_group(self, group_id):
        group_idx = self.group_id2group_idx(group_id)
        return len(self.grouped_idx_lists[group_idx])

    def sample_idx2group_id_and_idx_within_group(self, sample_idx):
        offset = 0
        for group_idx, group_id in enumerate(self.group_ids):
            if sample_idx < offset + self.nbr_samples_in_group(group_id):
                idx_within_group = sample_idx - offset
                return group_id, idx_within_group
            offset += self.nbr_samples_in_group(group_id)
        # Should not be able to happen:
        assert False

    def pack(self):
        u_grouped = [self.u[:, idx_list] for idx_list in self.grouped_idx_lists]
        U_grouped = [self.U[:, idx_list] for idx_list in self.grouped_idx_lists]
        sample_confidences = 0.25 / np.prod(self.b_per_sample, axis=0)
        sample_confidences_grouped = [sample_confidences[idx_list] for idx_list in self.grouped_idx_lists]
        group_confidences = [sample_confidences_in_group.max() for sample_confidences_in_group in sample_confidences_grouped]
        self._packed = {
            'u_grouped': u_grouped,
            'U_grouped': U_grouped,
            'sample_confidences': sample_confidences,
            'sample_confidences_grouped': sample_confidences_grouped,
            'group_confidences': group_confidences,
        }

    def add_correspondence_group(self, group_id, keypoint, u_unnorm, b):
        """
        Adds a group of unnormalized 2D points u corresponding to the given keypoint
        group_id            - int
        keypoint            - array (3,)
        u_unnorm            - array (2, N)
        b                   - array (2, N)
        """
        curr_nbr_samples = self.nbr_samples
        curr_nbr_groups = self.nbr_groups
        nbr_samples_added = u_unnorm.shape[1]
        u = normalize(pextend(u_unnorm), self.K)
        self.group_ids.append(group_id)
        self._group_id2group_idx_dict[group_id] = curr_nbr_groups
        self.grouped_idx_lists.append(list(range(curr_nbr_samples, curr_nbr_samples+nbr_samples_added)))
        self.u = np.concatenate([self.u, u], axis=1)
        self.b_per_sample = np.concatenate([self.b_per_sample, b], axis=1)
        self.U = np.concatenate([self.U, pextend(np.tile(keypoint[:,np.newaxis], (1, nbr_samples_added)))], axis=1)

def normalize_vec(vec):
    return vec / sum(vec)

class RansacEstimator():
    def __init__(self, corr_set, nransac=100, ransacthr=0.02, confidence_based_sampling=True):
        self.corr_set = corr_set
        self.corr_set.pack()
        self.nransac = nransac
        self.ransacthr = ransacthr
        self.confidence_based_sampling = confidence_based_sampling

    def _reproj_residuals(self, P):
        u_reproj = pflat(np.dot(P, self.corr_set.U))
        return u_reproj[0:2,:] - self.corr_set.u[:2,:]

    def _score_samples_reproj(self, P):
        return np.linalg.norm(self._reproj_residuals(P), axis=0)

    def _score_samples_nll(self, P):
        log_likelihoods = -np.abs(self._reproj_residuals(P)) / self.corr_set.b_per_sample - np.log(2*self.corr_set.b_per_sample)
        return -np.exp(np.sum(log_likelihoods, axis=0))

    def _score_samples_confidence_level(self, P):
        """
        Calculates the probability mass for the likelihood function of each sample in the x / y intervals [-res_x, res_x] / [-res_y, res_y].
        This probability mass corresponds to the confidence level required for the sample to be included in the corresponding confidence interval.
        """
        # Difference of CDFs in x & y direction
        probmass_coordwise = 1 - np.exp(-np.abs(self._reproj_residuals(P)) / self.corr_set.b_per_sample)
        probmass = np.prod(probmass_coordwise, axis=0)
        return probmass

    def evaluate_hypothesis(self, P):
        LIKELIHOOD_FLAG = False
        if not LIKELIHOOD_FLAG:
            # Reprojection error
            scores = self._score_samples_reproj(P)
            inlier_mask = scores < self.ransacthr
            if self.confidence_based_sampling:
                fraction_inliers = np.sum(self.corr_set.sample_confidences[inlier_mask]) / np.sum(self.corr_set.sample_confidences)
            else:
                fraction_inliers = np.sum(inlier_mask) / self.corr_set.nbr_samples
        else:
            # # Likelihood
            # self.ransacthr = -0.5
            # scores = self._score_samples_nll(P)
            # print(np.mean(scores))
            # inlier_mask = scores < self.ransacthr
            # fraction_inliers = np.sum(inlier_mask) / self.corr_set.nbr_samples

            # Likelihood probability mass at tails
            # self.ransacthr = 0.01 # 1 % confidence
            # self.ransacthr = 0.05 # 5 % confidence
            self.ransacthr = 0.3 # 30 % confidence
            scores = self._score_samples_confidence_level(P)
            inlier_mask = scores < self.ransacthr
            fraction_inliers = np.sum(inlier_mask) / self.corr_set.nbr_samples

        return inlier_mask, fraction_inliers

    def _sample_minimal_set(self):
        if self.confidence_based_sampling:
            selected_groups = np.random.choice(self.corr_set.nbr_groups, 3, p=normalize_vec(self.corr_set.group_confidences))
            return [np.random.choice(self.corr_set.grouped_idx_lists[group_idx], p=normalize_vec(self.corr_set.sample_confidences_grouped[group_idx])) for group_idx in selected_groups]
        else:
            selected_groups = np.random.choice(self.corr_set.nbr_groups, 3)
            return [np.random.choice(self.corr_set.grouped_idx_lists[group_idx]) for group_idx in selected_groups]

    def estimate(self):
        if self.corr_set.nbr_groups < 4:
            raise RANSACException("Found correspondence for {} keypoints only.".format(self.corr_set.nbr_groups))

        Pransac = None
        best_iteration = -1
        inlier_mask = np.zeros((self.corr_set.nbr_samples,), dtype=bool)
        fraction_inliers = 0
        for i in range(self.nransac):
            ind = self._sample_minimal_set()
            cameras = p3p_kneip(self.corr_set.u[:,ind], self.corr_set.U[:,ind])
            # cameras = resec3pts(self.corr_set.u[:,ind], self.corr_set.U[:,ind], coord_change=True)

            if i > best_iteration + 15 and best_iteration >= 0:
                # Stop prematurely if further improvement seems unlikely
                break
            for P in cameras:
                inlier_mask, curr_fraction_inliers = self.evaluate_hypothesis(P)
                if np.sum(inlier_mask) >= 4 and curr_fraction_inliers > fraction_inliers:
                    Pransac = P
                    fraction_inliers = curr_fraction_inliers
                    best_iteration = i

        # # EPnP on inliers
        # if np.sum(inlier_mask) < 4:
        #     fraction_inliers_epnp = 0
        # else:
        #     P_epnp = epnp(self.corr_set.u[:,inlier_mask], self.corr_set.U[:,inlier_mask])
        #     fraction_inliers_epnp = self.evaluate_hypothesis(P_epnp)[1] if P_epnp is not None else 0
        # 
        # # OpenGV RANSAC on all correspondences
        # P_opengv_ransac = opengv_ransac(self.corr_set.u, self.corr_set.U)
        # fraction_inliers_opengv_ransac = self.evaluate_hypothesis(P_opengv_ransac)[1] if P_opengv_ransac is not None else 0
        # 
        # print(fraction_inliers, fraction_inliers_opengv_ransac, fraction_inliers_epnp)

        if not fraction_inliers > 0:
            raise RANSACException("No inliers found")
        return Pransac, inlier_mask, fraction_inliers, i

class Runner(RunnerIf):
    def __init__(self, configs):
        super(Runner, self).__init__(configs, "rigidpose_ransac")
        self._class_map = get_class_map(configs)
        self._metadata = get_metadata(configs)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def _numpy2gpu(self, np_array):
        return torch.from_numpy(np_array).to(torch.device(get_device()))

    def _get_index_map(self, img_dims, stride):
        img_height, img_width = img_dims
        assert img_height % stride == 0
        assert img_width % stride == 0
        map_height = img_height // stride
        map_width = img_width // stride
        return stride * np.indices((map_height, map_width), dtype=np.float32)

    def run(self, cnn_outs, batch, frame_index):
        # Detach output tensors
        kp_maps_dict = {task_name: task_output[0][frame_index].detach() for task_name, task_output in cnn_outs.items() if task_name.startswith('keypoint')}
        kp_ln_b_maps_dict = {task_name: task_output[1][frame_index].detach() for task_name, task_output in cnn_outs.items() if task_name.startswith('keypoint')}
        visibility_maps = self._sigmoid(cnn_outs['clsnonmutex'][0][frame_index].detach())

        group_logit_maps = self._sigmoid(cnn_outs['clsgroup'][0][frame_index].detach())
        seg_map = group_logit_maps.argmax(0)
        gt_seg_map = batch.gt_map['clsgroup'][frame_index][0,:,:]

        # Suppress visibility maps outside of segmentation
        for group_id in self._class_map.get_group_ids():
            class_ids = np.array([self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx) for kp_idx in range(NBR_KEYPOINTS)])
            for kp_idx in range(NBR_KEYPOINTS):
                class_id = self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx)
                # visibility_maps[class_id-2,:,:][seg_map != group_id] = 0.0
                visibility_maps[class_id-2,:,:][gt_seg_map != group_id] = 0.0

        # Intrinsic camera parameters
        K = batch.calibration[frame_index][:,:3]

        # Generate index map
        index_map = self._numpy2gpu(self._get_index_map(self._configs.data.img_dims, self._configs.network.output_stride))

        frame_results = {}
        for group_id in self._class_map.get_group_ids():
            class_ids = [self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx) for kp_idx in range(NBR_KEYPOINTS)]
            group_label = self._class_map.group_label_from_group_id(group_id)

            corr_set = GroupedCorrespondenceSet(K)

            # For each keypoint, find and store samples from visible grid cells
            for kp_idx in range(NBR_KEYPOINTS):
                class_id = class_ids[kp_idx]
                key = '{}_{}'.format('keypoint', self._class_map.label_from_id(class_id))

                th = 0.5
                visibility_map = visibility_maps[class_id-2,:,:]
                mask_confident = visibility_map >= th
                nbr_confident = torch.sum(mask_confident).cpu().numpy()
                if nbr_confident == 0:
                    continue

                # visib_vec = visibility_map[mask_confident].flatten()
                # idx_x_vec = index_map[1,:,:][mask_confident].flatten()
                # idx_y_vec = index_map[0,:,:][mask_confident].flatten()
                kp_x_vec = (index_map[1,:,:] + kp_maps_dict[key]) [0,:,:][mask_confident].flatten()
                kp_y_vec = (index_map[0,:,:] + kp_maps_dict[key]) [1,:,:][mask_confident].flatten()
                kp_x_ln_b_vec = kp_ln_b_maps_dict[key][0,:,:][mask_confident].flatten()
                kp_y_ln_b_vec = kp_ln_b_maps_dict[key][1,:,:][mask_confident].flatten()
                # Laplace distribution, going from log(b) to b, to sigma=sqrt(2)*b
                # kp_std1_vec = math.sqrt(2) * torch.exp(kp_x_ln_b_vec)
                # kp_std2_vec = math.sqrt(2) * torch.exp(kp_y_ln_b_vec)

                MAX_NBR_SAMPLES = 100
                if nbr_confident > MAX_NBR_SAMPLES:
                    _, top_confident = torch.topk(kp_x_ln_b_vec+kp_y_ln_b_vec, k=MAX_NBR_SAMPLES, largest=False, sorted=False)
                    kp_x_vec = kp_x_vec[top_confident]
                    kp_y_vec = kp_y_vec[top_confident]
                    kp_x_ln_b_vec = kp_x_ln_b_vec[top_confident]
                    kp_y_ln_b_vec = kp_y_ln_b_vec[top_confident]

                # center_likelihood_vec = (0.5 / torch.exp(kp_x_ln_b_vec)) * (0.5 / torch.exp(kp_y_ln_b_vec))

                # Could not happen, right..?
                # if not center_likelihood_vec.max() > 0.0:
                #     print("Discarding group - no samples with confidence > 0.0")
                #     continue

                corr_set.add_correspondence_group(
                    kp_idx,
                    self._metadata['objects'][group_label]['keypoints'][:,kp_idx],
                    np.vstack([
                        kp_x_vec.cpu().numpy(),
                        kp_y_vec.cpu().numpy(),
                    ]),
                    np.vstack([
                        torch.exp(kp_x_ln_b_vec).cpu().numpy(),
                        torch.exp(kp_y_ln_b_vec).cpu().numpy(),
                    ]),
                    # center_likelihood_vec.cpu().numpy(),
                )

            ransac_estimator = RansacEstimator(
                corr_set,
                nransac=100,
                ransacthr=0.02,
                confidence_based_sampling=True,
            )
            try:
                P_ransac, inlier_set, fraction_inliers, niter = ransac_estimator.estimate()
                print("Object {}: Inlier fraction: {:.4f}, Iterations: {}".format(group_label, fraction_inliers, niter))
            except RANSACException as e:
                print("No solution found through RANSAC.")
                print(e)
                P_ransac = None
            if P_ransac is None:
                continue
            frame_results[group_id] = {
                'P_ransac': P_ransac,
            }

        return frame_results
