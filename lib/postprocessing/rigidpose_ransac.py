from attrdict import AttrDict

import torch

import numpy as np
import math

from lib.constants import NBR_KEYPOINTS, VISIB_TH
from lib.data import maps
from lib.utils import get_device, get_class_map, get_metadata
from lib.postprocessing import RunnerIf
from lib.rigidpose.pose_estimator import PoseEstimator, ransac, normalize, RANSACException, resec3pts, pflat, pextend
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
        self.U = np.empty((4, 0))
        self.u = np.empty((3, 0))
        self.sample_confidences = np.empty((0,))

    @property
    def nbr_samples(self):
        return self.u.shape[1]

    @property
    def nbr_groups(self):
        return len(self.grouped_idx_lists)

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

    def corr_idx2group_id_and_idx_within_group(self, corr_idx):
        offset = 0
        for group_idx, group_id in enumerate(self.group_ids):
            if corr_idx < offset + self.nbr_samples_in_group(group_id):
                idx_within_group = corr_idx - offset
                return group_id, idx_within_group
            offset += self.nbr_samples_in_group(group_id)
        # Should not be able to happen:
        assert False

    def pack(self):
        u_grouped = [self.u[:, idx_list] for idx_list in self.grouped_idx_lists]
        U_grouped = [self.U[:, idx_list] for idx_list in self.grouped_idx_lists]
        sample_confidences_grouped = [self.sample_confidences[idx_list] for idx_list in self.grouped_idx_lists]
        group_confidences = [sample_confidences_in_group.max() for sample_confidences_in_group in sample_confidences_grouped]
        self._packed = {
            'u_grouped': u_grouped,
            'U_grouped': U_grouped,
            'sample_confidences_grouped': sample_confidences_grouped,
            'group_confidences': group_confidences,
        }

    def add_correspondence_group(self, group_id, keypoint, u_unnorm, sample_confidences=None):
        """
        Adds a group of unnormalized 2D points u corresponding to the given keypoint
        group_id            - int
        keypoint            - array (3,)
        u_unnorm            - array (2, N)
        sample_confidences  - array (N,)
        """
        curr_nbr_samples = self.nbr_samples
        curr_nbr_groups = self.nbr_groups
        nbr_samples_added = u_unnorm.shape[1]
        if sample_confidences is None:
            sample_confidences = np.ones((nbr_samples_added,))
        u = normalize(pextend(u_unnorm), self.K)
        self.group_ids.append(group_id)
        self._group_id2group_idx_dict[group_id] = curr_nbr_groups
        self.grouped_idx_lists.append(list(range(curr_nbr_samples, curr_nbr_samples+nbr_samples_added)))
        self.u = np.concatenate([self.u, u], axis=1)
        self.sample_confidences = np.concatenate([self.sample_confidences, sample_confidences], axis=0)
        self.U = np.concatenate([self.U, pextend(np.tile(keypoint[:,np.newaxis], (1, nbr_samples_added)))], axis=1)

def normalize_vec(vec):
    return vec / sum(vec)

class RansacEstimator():
    def __init__(self, configs, corr_set, nransac=100, ransacthr=0.02, confidence_based_sampling=True):
        self._configs = configs
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
        best_minimal_set = None
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
                    best_minimal_set = ind
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
        return Pransac, best_minimal_set, inlier_mask, fraction_inliers, i

class LevenbergMarquardtJonBarronEstimator():
    def __init__(
            self,
            configs,
            corr_set,
            P0=None,
        ):
        self._configs = configs
        self.corr_set = corr_set
        self.corr_set.pack()
        self.P0 = P0
        self.pose_estimator = self._initialize_pose_estimator()

    def _initialize_pose_estimator(self):
        c = 65.0 # Ad-hoc - looked similar to mixture of laplacian PDFs, when plotting exp(-sum_of_jb_losses_for_residuals)
        K = self.corr_set.K
        U0 = pflat(self.corr_set.U)[:3,:]
        u_unnorm = pflat(K @ self.corr_set.u)[:2,:]
        Uanno0 = None # Model vertices not needed for forward pass
        # Uanno0 = ...
        w = None
        # w = 

        return PoseEstimator(
            self._configs,
            K, # Calibration
            U0, # 3D correspondence points
            u_unnorm, # 2D correspondence pixels
            Uanno0=Uanno0, # Mesh vertices
            verbose=0,
            w=w,
            alpha_rho=-1.0,
            c=c,
            lambda0=1e5,
            max_refine_iters=200,
        )

    def _reproj_residuals(self, P):
        u_reproj = pflat(np.dot(P, self.corr_set.U))
        return u_reproj[0:2,:] - self.corr_set.u[:2,:]

    def estimate(self):
        if self.corr_set.nbr_groups < 4:
            raise RANSACException("Found correspondence for {} keypoints only.".format(self.corr_set.nbr_groups))

        R_pred, t_pred, rhototal, drhototal_dx, final_res, refine_iters, final_lambda_refinement = self.pose_estimator.pose_forward(P0 = self.P0)
        P_est = np.concatenate([R_pred, t_pred], axis=1)

        return P_est

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
        # calib = batch.calibration[frame_index]
        # # image_tensor = normalize(batch.input[frame_index], mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        # image_tensor = batch.input[frame_index] 
        # # frame_id = batch.id[frame_index]
        # annotations = batch.annotation[frame_index]

        detections = []
        for obj in zip(*cnn_outs[frame_index].values()):
            obj = dict(zip(cnn_outs[frame_index], obj))
            if obj['scores'] < 0.1:
                continue
            detections.append(AttrDict({
                'bbox2d': obj['boxes'],
                'cls': obj['labels'].item(),
                'confidence': obj['scores'],
                'keypoints': obj['keypoints'].cpu().detach().numpy(),
                'pose_results': None,
            }))

        # gt_visibility_maps = batch.gt_map['clsnonmutex'][frame_index]
        # gt_seg_map = batch.gt_map['clsgroup'][frame_index][0,:,:]

        # GT annotations
        if self._configs.postprocessing.rigidpose_ransac.method == 'lm_jb' and self._configs.postprocessing.rigidpose_ransac.lm_jb.initialization == 'gt':
            # assert mode in (TRAIN, VAL):

            # Pick one sample from batch of ground truth annotations
            annotations = batch.annotation[frame_index]

            anno_lookup = dict(zip([anno.cls for anno in annotations], annotations))

        # Intrinsic camera parameters
        K = batch.calibration[frame_index][:,:3]

        frame_results = detections
        for det in detections:
            class_id = det.cls
            class_label = self._class_map.label_from_id(class_id)

            curr_results = {}
            det.pose_results = curr_results

            corr_set = GroupedCorrespondenceSet(K)

            curr_results['keypoints'] = [None]*NBR_KEYPOINTS
            # For each keypoint, find and store samples from visible grid cells
            for kp_idx in range(NBR_KEYPOINTS):
                kp_visible_flag = det.keypoints[kp_idx,2] > 0.5
                print(kp_visible_flag)
                if kp_visible_flag:
                    x, y = det.keypoints[kp_idx,:2]
                    corr_set.add_correspondence_group(
                        kp_idx,
                        self._metadata['objects'][class_label]['keypoints'][:,kp_idx],
                        det.keypoints[[kp_idx],:2].T,
                    )

            curr_results['corr_set'] = corr_set

            if self._configs.postprocessing.rigidpose_ransac.method == 'ransac':
                ransac_estimator = RansacEstimator(
                    self._configs,
                    corr_set,
                    nransac=self._configs.postprocessing.rigidpose_ransac.ransac.n_iter,
                    ransacthr=self._configs.postprocessing.rigidpose_ransac.ransac.th,
                    confidence_based_sampling=True,
                )
                try:
                    P_ransac, indices_best_minimal_set, inlier_mask, fraction_inliers, niter = ransac_estimator.estimate()
                    print("Object {}: Inlier fraction: {:.4f}, Iterations: {}".format(class_label, fraction_inliers, niter))
                except RANSACException as e:
                    print("No solution found through RANSAC.")
                    print(e)
                    P_ransac = None

                if P_ransac is None:
                    curr_results['P_est'] = None
                    curr_results['ransac'] = None
                    continue
                best_minimal_set = {}
                for corr_idx in indices_best_minimal_set:
                    curr_kp_idx, corr_idx_within_kp_group = corr_set.corr_idx2class_id_and_idx_within_group(corr_idx)
                    # best_minimal_set[curr_kp_idx] = corr_idx
                    best_minimal_set[curr_kp_idx] = corr_idx_within_kp_group
                curr_results['P_est'] = P_ransac
                curr_results['ransac'] = {
                    'P': P_ransac,
                    'fraction_inliers': fraction_inliers,
                    'inlier_mask': inlier_mask,
                    'best_minimal_set': best_minimal_set,
                }
            elif self._configs.postprocessing.rigidpose_ransac.method == 'lm_jb':
                if self._configs.postprocessing.rigidpose_ransac.lm_jb.initialization == 'gt':
                    if class_id in anno_lookup:
                        P0 = np.concatenate([anno_lookup[class_id].rotation, anno_lookup[class_id].location[:,None]], axis=1)
                    else:
                        P0 = None
                elif self._configs.postprocessing.rigidpose_ransac.lm_jb.initialization == 'ransac':
                    # Not implemented
                    # P0 = 
                    assert False
                else:
                    assert False
                if P0 is not None and corr_set.nbr_groups >= 4:
                    lm_jb_estimator = LevenbergMarquardtJonBarronEstimator(
                        self._configs,
                        corr_set,
                        P0=P0,
                    )
                    P_est = lm_jb_estimator.estimate()
                else:
                    P_est = None
                curr_results['P_est'] = P_est
                curr_results['ransac'] = None
            else:
                assert False


        return frame_results
