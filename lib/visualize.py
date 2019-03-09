"""Visualizer."""
from os.path import join
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot, patches

import numpy as np

from torch import nn
from torchvision.transforms.functional import normalize
from tensorboardX import SummaryWriter

from lib.constants import PYPLOT_DPI, BOX_SKELETON, CORNER_COLORS, NBR_KEYPOINTS, PATCH_SIZE, GT_TYPE, CNN_TYPE, DET_TYPE
from lib.constants import TV_MEAN, TV_STD
from lib.utils import project_3d_pts, construct_3d_box, get_metadata, get_class_map

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

    def report_loss(self, epoch, losses, mode):
        self._writer.add_scalar('loss/{}'.format(mode), sum(losses.values()), epoch)
        self._writer.add_scalars('task_losses/{}'.format(mode), losses, epoch)

    def report_score(self, epoch, score, mode):
        self._writer.add_scalar('score/{}'.format(mode), score, epoch)

    def save_images(self, batch, cnn_outs, output, mode, index, sample=-1):
        if not any(self._configs.visualization.values()):
            return
        calib = batch.calibration[sample]
        image_tensor = normalize(batch.input[sample], mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        frame_id = batch.id[sample]

        # # Pick one sample from batch of detections / whatever comes from postprocessing modules
        # detections = output[frame_id]

        # Pick one sample from batch of ground truth annotations
        annotations = batch.annotation[sample]

        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

        def tensor2numpy(tensor, sample_idx, upsample_and_permute=False):
            tensor = tensor.detach()
            if upsample_and_permute:
                tensor = nn.Upsample(size=self._configs.data.img_dims,mode='nearest')(tensor)
            tensor = tensor[sample_idx,:,:,:]
            array = tensor.cpu().numpy()
            # if upsample_and_permute:
            #     array = np.moveaxis(array, 0, -1)
            return array

        # Pick one sample from batch of output feature maps
        cnn_outs_task, cnn_outs_ln_b = cnn_outs
        kp_maps_dict = {task_name: tensor2numpy(tensor, sample, upsample_and_permute=False) for task_name, tensor in cnn_outs_task.items() if task_name.startswith('keypoint')}
        kp_confidence_maps_dict = {task_name: tensor2numpy(tensor, sample, upsample_and_permute=False) for task_name, tensor in cnn_outs_ln_b.items() if task_name.startswith('keypoint')}
        visibility_maps_lowres = sigmoid(tensor2numpy(cnn_outs_task['clsnonmutex'], sample))
        visibility_maps_highres = sigmoid(tensor2numpy(cnn_outs_task['clsnonmutex'], sample, upsample_and_permute=True))


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

        img = np.moveaxis(image_tensor.numpy(), 0, -1)
        # img = grayscale2rgb(rgb2grayscale(rgb))

        nrows = NBR_KEYPOINTS + 1
        ncols = 2

        figwidth_fullres = ncols*img.shape[1] / PYPLOT_DPI
        figheight_fullres = nrows*img.shape[0] / PYPLOT_DPI

        # figwidth = figwidth_fullres
        # figheight = figheight_fullres
        figwidth = 10
        figheight = figheight_fullres * (figwidth / figwidth_fullres)

        def plot_img(axes, img, title):
            axes.axis('off')
            axes.imshow(img)
            axes.set_title(title)

        anno_lookup = dict(zip([anno.cls for anno in annotations], annotations))

        for group_id in self._class_map.get_group_ids():
            class_ids = [self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx) for kp_idx in range(NBR_KEYPOINTS)]

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
            plot_img(axes_array[0,0], img, 'Image')
            for kp_idx in range(NBR_KEYPOINTS):
                class_id = class_ids[kp_idx]
                kp_color = np.array(pyplot.cm.tab20.colors[kp_idx])
                # heatmap_color = kp_color
                heatmap_color = np.array([1.0, 0.0, 1.0])

                visibility_map_highres = visibility_maps_highres[class_id-2,:,:]

                lambda_map = 0.6*visibility_map_highres
                heatmap = blend_rgb(img, get_uniform_color(heatmap_color), lambda_map)
                heatmap[:100,:100,:] = kp_color
                plot_img(axes_array[kp_idx+1,0], heatmap, 'Keypoint {:02d}'.format(kp_idx))

                plot_img(axes_array[kp_idx+1,1], img, 'Keypoint {:02d}'.format(kp_idx))

                if class_id in anno_lookup:
                    gt_color = 'red' if anno_lookup[class_id].self_occluded or anno_lookup[class_id].occluded else 'blue'
                    axes_array[kp_idx+1,1].add_patch(patches.Circle(anno_lookup[class_id].keypoint, radius=4, color=gt_color, edgecolor='black'))
            # pyplot.subplots_adjust(
            #     left = 0.0,
            #     right = 1.0,
            #     bottom = 0.0,
            #     top = 1.0,
            #     wspace = 0.0,
            #     hspace = 0.0,
            # )
            self._writer.add_figure('{}_{}'.format(mode, process_group_label(self._class_map.group_label_from_group_id(group_id))), fig, index)
