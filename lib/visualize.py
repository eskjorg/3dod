"""Visualizer."""
from os.path import join
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot, patches

import math
import numpy as np

import torch
from torch import nn
from torchvision.transforms.functional import normalize
from tensorboardX import SummaryWriter

from lib.constants import PYPLOT_DPI, BOX_SKELETON, CORNER_COLORS, NBR_KEYPOINTS, PATCH_SIZE, GT_TYPE, CNN_TYPE, DET_TYPE
from lib.constants import TV_MEAN, TV_STD
from lib.constants import TRAIN, VAL
from lib.utils import project_3d_pts, construct_3d_box, get_metadata, get_class_map
from lib.rigidpose.pose_estimator import pflat


class Visualizer:
    """Visualizer."""
    def __init__(self, configs):
        self._configs = configs
        self._metadata = get_metadata(self._configs)
        self._class_map = get_class_map(configs)
        vis_path = join(configs.experiment_path, 'visual')
        shutil.rmtree(vis_path, ignore_errors=True)
        self._writer = SummaryWriter(vis_path)
        self._corner_colors = CORNER_COLORS
        self._loss_count_dict = {'train': 0, 'val': 0}

    def report_loss(self, losses, mode):
        self._writer.add_scalar('loss/{}'.format(mode), sum(losses.values()), self._loss_count_dict[mode])
        self._writer.add_scalars('task_losses/{}'.format(mode), losses, self._loss_count_dict[mode])
        self._loss_count_dict[mode] += 1

    def report_score(self, epoch, score, mode):
        self._writer.add_scalar('score/{}'.format(mode), score, epoch)

    def plot_bbox3d(self, ax, K, R, t, x_bounds, y_bounds, z_bounds, **kwargs):
        if len(t.shape) == 1:
            t = t[:,None]
        corners = np.zeros((3, 8))

        corners[:,0] = [x_bounds[1], y_bounds[1], z_bounds[1]]
        corners[:,1] = [x_bounds[0], y_bounds[1], z_bounds[1]]
        corners[:,2] = [x_bounds[0], y_bounds[0], z_bounds[1]]
        corners[:,3] = [x_bounds[1], y_bounds[0], z_bounds[1]]
        corners[:,4] = [x_bounds[1], y_bounds[1], z_bounds[0]]
        corners[:,5] = [x_bounds[0], y_bounds[1], z_bounds[0]]
        corners[:,6] = [x_bounds[0], y_bounds[0], z_bounds[0]]
        corners[:,7] = [x_bounds[1], y_bounds[0], z_bounds[0]]

        corner_colors = ['k', 'w', 'r', 'g', 'y', 'c', 'm', 'b']

        U = pflat(K @ (R @ corners + t))
        xs = np.array([[U[0,0], U[0, 1], U[0, 2], U[0,3], U[0,4], U[0,5], U[0,6], U[0,7], U[0,0], U[0,1], U[0,2], U[0,3]], [U[0,1], U[0, 2], U[0,3], U[0,0], U[0,5], U[0,6], U[0,7], U[0,4], U[0,4], U[0,5], U[0,6], U[0,7]]])
        ys = np.array([[U[1,0], U[1,1], U[1,2], U[1,3], U[1,4], U[1,5], U[1,6], U[1,7], U[1,0], U[1,1], U[1,2], U[1,3]], [U[1,1], U[1,2], U[1,3], U[1,0], U[1,5], U[1,6], U[1,7], U[1,4], U[1,4], U[1,5], U[1,6], U[1,7]]])
        ax.plot([xs[0,:], xs[1,:]], [ys[0,:], ys[1,:]], **kwargs)
        corner_plot_args = []
        for pt, col in zip(U.T, corner_colors):
            corner_plot_args += [pt[0], pt[1], col+'.']
        ax.plot(*corner_plot_args)

    def save_images(self, batch, cnn_outs, output, mode, index, sample=-1):
        if not self._configs.visualization.keypoints:
            return

        K = batch.calibration[sample][:,:3]
        image_tensor = normalize(batch.input[sample], mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        frame_id = batch.id[sample]

        # Pick one sample from batch of detections / whatever comes from postprocessing modules
        detections = output[frame_id]

        if mode in (TRAIN, VAL):
            # Pick one sample from batch of ground truth annotations
            annotations = batch.annotation[sample]
        else:
            annotations = []

        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

        def tensor2numpy(tensor, sample_idx, upsample_and_permute=False):
            tensor = tensor.detach()
            if upsample_and_permute:
                if tensor.dtype == torch.long:
                    tensor = nn.Upsample(size=self._configs.data.img_dims,mode='nearest')(tensor.double()).long()
                else:
                    tensor = nn.Upsample(size=self._configs.data.img_dims,mode='nearest')(tensor)
            tensor = tensor[sample_idx,:,:,:]
            array = tensor.cpu().numpy()
            # if upsample_and_permute:
            #     array = np.moveaxis(array, 0, -1)
            return array

        # Pick one sample from batch of output feature maps
        kp_maps_dict = {task_name: tensor2numpy(task_output[0], sample, upsample_and_permute=False) for task_name, task_output in cnn_outs.items() if task_name.startswith('keypoint')}
        kp_ln_b_maps_dict = {task_name: tensor2numpy(task_output[1], sample, upsample_and_permute=False) for task_name, task_output in cnn_outs.items() if task_name.startswith('keypoint')}
        visibility_maps_lowres = sigmoid(tensor2numpy(cnn_outs['clsnonmutex'][0], sample))
        visibility_maps_highres = sigmoid(tensor2numpy(cnn_outs['clsnonmutex'][0], sample, upsample_and_permute=True))
        group_logit_maps_lowres = sigmoid(tensor2numpy(cnn_outs['clsgroup'][0], sample))
        group_logit_maps_highres = sigmoid(tensor2numpy(cnn_outs['clsgroup'][0], sample, upsample_and_permute=True))
        seg_map_lowres = group_logit_maps_lowres.argmax(0)
        seg_map_highres = group_logit_maps_highres.argmax(0)

        if mode in (TRAIN, VAL):
            # And corresponding ground truth
            gt_visibility_maps_highres = tensor2numpy(batch.gt_map['clsnonmutex'], sample, upsample_and_permute=True)
            gt_seg_map_lowres = tensor2numpy(batch.gt_map['clsgroup'], sample)[0,:,:]
            gt_seg_map_highres = tensor2numpy(batch.gt_map['clsgroup'], sample, upsample_and_permute=True)[0,:,:]

        # Index map
        def get_index_map(img_dims, stride):
            img_height, img_width = img_dims
            assert img_height % stride == 0
            assert img_width % stride == 0
            map_height = img_height // stride
            map_width = img_width // stride
            return stride * np.indices((map_height, map_width), dtype=np.float32)
        index_map_lowres = get_index_map(self._configs.data.img_dims, self._configs.network.output_stride)
        index_map_highres = get_index_map(self._configs.data.img_dims, 1)

        def blend_rgb(rgb1, rgb2, lambda_map):
            """
            lambda = 0 - rgb1 unchanged
            lambda = 1 - rgb2 unchanged
            """
            blended =      lambda_map[:,:,np.newaxis]  * rgb2 + \
                    (1.0 - lambda_map[:,:,np.newaxis]) * rgb1
            return blended

        def get_uniform_color(color):
            return color[np.newaxis,np.newaxis,:]

        def rgb2grayscale(rgb):
            return rgb.prod(axis=2)

        def grayscale2rgb(grayscale):
            return np.tile(grayscale[:,:,np.newaxis], (1,1,3))

        def expand_bbox(bbox2d, resize_factor):
            x1, y1, x2, y2 = bbox2d
            center_x = 0.5*(x1+x2)
            center_y = 0.5*(y1+y2)
            return (
                max(-0.5,                                  center_x + resize_factor*(x1 - center_x)),
                max(-0.5,                                  center_y + resize_factor*(y1 - center_y)),
                min(-0.5 + self._configs.data.img_dims[1], center_x + resize_factor*(x2 - center_x)),
                min(-0.5 + self._configs.data.img_dims[0], center_y + resize_factor*(y2 - center_y)),
            )

        img = np.moveaxis(image_tensor.numpy(), 0, -1)
        # img = grayscale2rgb(rgb2grayscale(rgb))

        def plot_img(ax, img, title, bbox2d=None):
            img = np.clip(img, 0.0, 1.0)
            if bbox2d is None:
                ax.axis('on')
                ax.set_xlim(-0.5,                                  -0.5 + self._configs.data.img_dims[1])
                ax.set_ylim(-0.5 + self._configs.data.img_dims[0], -0.5)
            else:
                x1, y1, x2, y2 = bbox2d
                ax.set_xlim(x1, x2)
                ax.set_ylim(y2, y1)
            ax.autoscale(enable=False)
            ax.imshow(img)
            ax.set_title(title)

        anno_lookup = dict(zip([anno.cls for anno in annotations], annotations))
        anno_group_lookup = {}
        for group_id in self._class_map.get_group_ids():
            # All keypoints share the same pose annotation - choose the first existing one
            for kp_idx in range(NBR_KEYPOINTS):
                class_id = self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx)
                if class_id in anno_lookup:
                    anno_group_lookup[group_id] = anno_lookup[class_id]
                    break

        def plot_poses(ax, group_ids, annotation, detection):
            for group_id in group_ids:
                group_label = self._class_map.group_label_from_group_id(group_id)
                if group_id in anno_group_lookup:
                    anno = anno_group_lookup[group_id]
                    self.plot_bbox3d(ax, K, anno.rotation, anno.location.numpy(), *self._metadata['objects'][group_label]['bbox3d'], color='b', linestyle='-', linewidth=1)
                if group_id in detections:
                    det = detections[group_id]
                    if det['ransac'] is not None:
                        self.plot_bbox3d(ax, K, det['ransac']['P'][:,:3], det['ransac']['P'][:,3], *self._metadata['objects'][group_label]['bbox3d'], color='g', linestyle=':', linewidth=1)

        fig, axes_array = pyplot.subplots(
            nrows=1,
            ncols=1,
            figsize=[img.shape[1] / PYPLOT_DPI, img.shape[0] / PYPLOT_DPI],
            squeeze=False,
            dpi=PYPLOT_DPI,
            tight_layout=True,
        )
        plot_img(axes_array[0,0], img, 'Pose')
        plot_poses(axes_array[0,0], self._class_map.get_group_ids(), annotations, detections)
        self._writer.add_figure(mode, fig, index)

        nrows = NBR_KEYPOINTS + 1
        ncols = 4

        figwidth_fullres = ncols*img.shape[1] / PYPLOT_DPI
        figheight_fullres = nrows*img.shape[0] / PYPLOT_DPI

        # figwidth = figwidth_fullres
        # figheight = figheight_fullres
        figwidth = 15
        figheight = figheight_fullres * (figwidth / figwidth_fullres)


        # Suppress visibility maps outside of segmentation
        for group_id in self._class_map.get_group_ids():
            # print(type(visibility_maps_lowres))
            # print(type(seg_map_lowres != group_id))
            # print(type(seg_map_lowres))
            # 
            # print(visibility_maps_lowres[class_ids-2,:,:][np.tile((seg_map_lowres[np.newaxis,:,:] != group_id), (NBR_KEYPOINTS, 1, 1))])
            # print(visibility_maps_lowres[class_ids-2,:,:][np.tile((seg_map_lowres[np.newaxis,:,:] != group_id), (NBR_KEYPOINTS, 1, 1))].shape)
            # visibility_maps_lowres[class_ids-2,:,:][np.tile((seg_map_lowres[np.newaxis,:,:] != group_id), (NBR_KEYPOINTS, 1, 1))] = 0.0
            # print(visibility_maps_lowres[class_ids-2,:,:][np.tile((seg_map_lowres[np.newaxis,:,:] != group_id), (NBR_KEYPOINTS, 1, 1))])
            # visibility_maps_highres[class_ids-2,:,:][np.tile((seg_map_highres[np.newaxis,:,:] != group_id), (NBR_KEYPOINTS, 1, 1))] = 0.0

            class_ids = np.array([self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx) for kp_idx in range(NBR_KEYPOINTS)])
            for kp_idx in range(NBR_KEYPOINTS):
                class_id = self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx)
                # visibility_maps_lowres[class_id-2,:,:][seg_map_lowres != group_id] = 0.0
                # visibility_maps_highres[class_id-2,:,:][seg_map_highres != group_id] = 0.0
                visibility_maps_lowres[class_id-2,:,:][gt_seg_map_lowres != group_id] = 0.0
                visibility_maps_highres[class_id-2,:,:][gt_seg_map_highres != group_id] = 0.0

        for group_id in self._class_map.get_group_ids():
            class_ids = [self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx) for kp_idx in range(NBR_KEYPOINTS)]
            group_label = self._class_map.group_label_from_group_id(group_id)

            fig, axes_array = pyplot.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=[figwidth, figheight],
                squeeze=False,
                dpi=PYPLOT_DPI,
                tight_layout=True,
            )
            def process_group_label(label):
                lookup = {
                    '01': 'ape',
                    '05': 'can',
                    '06': 'cat',
                    '08': 'driller',
                    '09': 'duck',
                    '10': 'eggbox',
                    '11': 'glue',
                    '12': 'holepuncher',
                }
                return lookup[label]

            bbox2d = expand_bbox(anno_group_lookup[group_id].bbox2d, 3.0) if group_id in anno_group_lookup else None

            # Regular size image + pose
            plot_img(axes_array[0,0], img, 'Pose')
            plot_poses(axes_array[0,0], [group_id], annotations, detections)

            # Close-up image + pose
            plot_img(axes_array[0,1], img, 'Pose close-up', bbox2d=bbox2d)
            plot_poses(axes_array[0,1], [group_id], annotations, detections)

            # Segmentation
            heatmap_color = np.array([1.0, 0.0, 1.0])
            heatmap = blend_rgb(img, get_uniform_color(heatmap_color), seg_map_highres == group_id)
            plot_img(axes_array[0,2], heatmap, 'Segmentation output')

            # BG-Segmentation
            heatmap_color = np.array([1.0, 0.0, 1.0])
            heatmap = blend_rgb(img, get_uniform_color(heatmap_color), gt_seg_map_highres == group_id)
            plot_img(axes_array[0,3], heatmap, 'GT Segmentation')

            for kp_idx in range(NBR_KEYPOINTS):
                class_id = class_ids[kp_idx]
                kp_color = np.array(pyplot.cm.tab20.colors[kp_idx])
                # heatmap_color = kp_color
                heatmap_color = np.array([1.0, 0.0, 1.0])


                if mode in (TRAIN, VAL):
                    # GT heatmap
                    gt_visibility_map_highres = gt_visibility_maps_highres[class_id-2,:,:]
                    lambda_map = 0.6*gt_visibility_map_highres
                    heatmap = blend_rgb(img, get_uniform_color(heatmap_color), lambda_map)
                    heatmap[:100,:100,:] = kp_color
                    plot_img(axes_array[kp_idx+1,0], heatmap, 'Keypoint {:02d} - GT'.format(kp_idx), bbox2d=bbox2d)

                    # GT position
                    if class_id in anno_lookup:
                        gt_color = 'red' if anno_lookup[class_id].self_occluded or anno_lookup[class_id].occluded else 'blue'
                        axes_array[kp_idx+1,0].add_patch(patches.Circle(anno_lookup[class_id].keypoint, radius=4, color=gt_color, edgecolor='black'))



                # Pred heatmap
                visibility_map_highres = visibility_maps_highres[class_id-2,:,:]
                lambda_map = 0.6*visibility_map_highres
                heatmap = blend_rgb(img, get_uniform_color(heatmap_color), lambda_map)
                heatmap[:100,:100,:] = kp_color
                plot_img(axes_array[kp_idx+1,1], heatmap, 'Keypoint {:02d} - Pred'.format(kp_idx), bbox2d=bbox2d)

                # Pred position
                likelihood_map = np.zeros(self._configs.data.img_dims)

                th = 0.5
                visibility_map_lowres = visibility_maps_lowres[class_id-2,:,:]
                mask_confident = visibility_map_lowres >= th
                nbr_confident = np.sum(mask_confident)
                if nbr_confident > 0:
                    visib_vec = visibility_map_lowres[mask_confident].flatten()
                    key = '{}_{}'.format('keypoint', self._class_map.label_from_id(class_id))
                    idx_x_vec = index_map_lowres[1,:,:][mask_confident].flatten()
                    idx_y_vec = index_map_lowres[0,:,:][mask_confident].flatten()
                    kp_x_vec = idx_x_vec + kp_maps_dict[key][0,:,:][mask_confident].flatten()
                    kp_y_vec = idx_y_vec + kp_maps_dict[key][1,:,:][mask_confident].flatten()
                    kp_x_ln_b_vec = kp_ln_b_maps_dict[key][0,:,:][mask_confident].flatten()
                    kp_y_ln_b_vec = kp_ln_b_maps_dict[key][1,:,:][mask_confident].flatten()
                    # Laplace distribution, going from log(b) to b, to sigma=sqrt(2)*b
                    kp_std1_vec = np.sqrt(2) * np.exp(kp_x_ln_b_vec)
                    kp_std2_vec = np.sqrt(2) * np.exp(kp_y_ln_b_vec)

                    nbr_sampled = min(15, nbr_confident)

                    # # Sample based on estimated visibility
                    # p = visib_vec / np.sum(visib_vec)

                    # Sample based on confidence
                    kp_avg_std_vec = 0.5*sum([kp_std1_vec, kp_std2_vec])
                    center_likelihood_vec = (0.5 / np.exp(kp_x_ln_b_vec)) * (0.5 / np.exp(kp_y_ln_b_vec))
                    p = center_likelihood_vec / np.sum(center_likelihood_vec)

                    # for idx in {np.argmin(0.5*sum([kp_std1_vec, kp_std2_vec]))}:
                    idx_sampled = np.random.choice(nbr_confident, nbr_sampled, p=p)
                    for idx in idx_sampled:
                        avg_std = 0.5*sum([kp_std1_vec[idx], kp_std2_vec[idx]])
                        nbr_std_px_half_faded = 5.0 # When std amounts to this number of pixels, KP color will be faded to half intensity
                        confidence_interp_factor = 1.0 / (1.0 + (avg_std/nbr_std_px_half_faded)**2)

                        # confidence_interp_factor = center_likelihood_vec[idx]

                        color = confidence_interp_factor * np.array([0.0, 1.0, 0.0]) + (1.0 - confidence_interp_factor) * np.array([0.0, 0.0, 0.0])
                        axes_array[kp_idx+1,1].plot([idx_x_vec[idx], kp_x_vec[idx]], [idx_y_vec[idx], kp_y_vec[idx]], '-', color=color)
                        # axes_array[kp_idx+1,1].plot(
                        #     [kp_x_vec[idx] - 0.5*kp_std1_vec[idx],    kp_x_vec[idx] + 0.5*kp_std1_vec[idx]],
                        #     [kp_y_vec[idx],                            kp_y_vec[idx]],
                        #     '-',
                        #     color='red',
                        # )
                        # axes_array[kp_idx+1,1].plot(
                        #     [kp_x_vec[idx],                            kp_x_vec[idx]],
                        #     [kp_y_vec[idx] - 0.5*kp_std2_vec[idx],    kp_y_vec[idx] + 0.5*kp_std2_vec[idx]],
                        #     '-',
                        #     color='red',
                        # )
                        axes_array[kp_idx+1,1].add_patch(patches.Circle([kp_x_vec[idx], kp_y_vec[idx]], radius=4, color=color, edgecolor='black'))

                    for idx in range(nbr_confident):
                        x = kp_x_vec[idx]
                        y = kp_y_vec[idx]
                        std1 = kp_std1_vec[idx]
                        std2 = kp_std2_vec[idx]
                        b1 = std1 / np.sqrt(2)
                        b2 = std2 / np.sqrt(2)

                        x1 = max(0, math.floor(x - 2.0*std1))
                        x2 = min(self._configs.data.img_dims[1], math.ceil(x + 2.0*std1))
                        y1 = max(0, math.floor(y - 2.0*std2))
                        y2 = min(self._configs.data.img_dims[0], math.ceil(y + 2.0*std2))

                        likelihood_map[y1:y2, x1:x2] += visib_vec[idx]*np.exp(sum([
                            -np.abs(index_map_highres[1, y1:y2, x1:x2] - x)/b1 - np.log(2*b1),
                            -np.abs(index_map_highres[0, y1:y2, x1:x2] - y)/b2 - np.log(2*b2),
                        ]))

                    likelihood_map /= nbr_confident

                # Plot likelihood values, to get a feel for its shape
                axes_array[kp_idx+1,3].plot(list(reversed(sorted(likelihood_map.flatten())))[:10000])
                axes_array[kp_idx+1,3].set_ylim(0, 0.001)

                # Uncertainty map
                # lambda_map = 0.6*likelihood_map
                lambda_map = likelihood_map / likelihood_map.max()
                # lambda_map = 10.0*np.clip(likelihood_map/10.0, 0.0, 1.0)
                # lambda_map = np.clip(likelihood_map/0.7, 0.0, 1.0)
                heatmap = blend_rgb(img, get_uniform_color(heatmap_color), lambda_map)
                heatmap[:100,:100,:] = kp_color
                # plot_img(axes_array[kp_idx+1,2], heatmap, 'Keypoint {:02d} - Uncertainty'.format(kp_idx), bbox2d=bbox2d)
                plot_img(axes_array[kp_idx+1,2], heatmap, 'Keypoint {:02d} - Uncertainty'.format(kp_idx), bbox2d=None)

                if detections[group_id]['ransac'] is not None and kp_idx in detections[group_id]['ransac']['best_minimal_set']:
                    corr_idx_within_kp_group = detections[group_id]['ransac']['best_minimal_set'][kp_idx]
                    x = detections[group_id]['keypoints'][kp_idx]['kp_x_vec'][corr_idx_within_kp_group]
                    y = detections[group_id]['keypoints'][kp_idx]['kp_y_vec'][corr_idx_within_kp_group]
                    # print(kp_idx, x, y)
                    # axes_array[kp_idx+1,2].add_patch(patches.Circle([x, y], radius=10, color='red', edgecolor='black'))
                    axes_array[kp_idx+1,2].plot([x], [y], 'x', markersize=10, color='green')
                else:
                    # print(kp_idx, False, False)
                    pass






            # pyplot.subplots_adjust(
            #     left = 0.0,
            #     right = 1.0,
            #     bottom = 0.0,
            #     top = 1.0,
            #     wspace = 0.0,
            #     hspace = 0.0,
            # )
            self._writer.add_figure('{}_{}'.format(mode, process_group_label(group_label)), fig, index)
