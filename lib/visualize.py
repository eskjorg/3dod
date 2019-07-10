"""Visualizer."""
from os.path import join
import shutil

from attrdict import AttrDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot, patches

import numpy as np

from torchvision.transforms.functional import normalize
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

from lib.constants import PYPLOT_DPI, BOX_SKELETON, CORNER_COLORS, KEYPOINT_COLORS
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
        self._keypoint_colors = KEYPOINT_COLORS
        self._loss_count_dict = {'train': 0, 'val': 0}

    def report_loss(self, losses, mode):
        self._writer.add_scalar('loss/{}'.format(mode), sum(losses.values()), self._loss_count_dict[mode])
        self._writer.add_scalars('task_losses/{}'.format(mode), losses, self._loss_count_dict[mode])
        self._loss_count_dict[mode] += 1

    def report_score(self, epoch, score, mode):
        self._writer.add_scalar('score/{}'.format(mode), score, epoch)

    def save_images(self, batch, cnn_outs, frame_results, mode, index, sample=-1):
        if not any(self._configs.visualization.values()):
            return
        calib = batch.calibration[sample]
        # image_tensor = normalize(batch.input[sample], mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        image_tensor = batch.input[sample] 
        # frame_id = batch.id[sample]
        annotations = batch.annotation[sample]

        frame_id = batch.id[sample]
        detections = frame_results[frame_id]

        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # Visualize estimated pose
        # Visualize estimated pose
        # Visualize estimated pose
        # Visualize estimated pose
        # Visualize estimated pose
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # for det in detections:
        #     print(det.keypoints)
        #     print(det.pose_results)

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

        def plot_gt(axes):
            for feature in self._configs.visualization.gt:
                for annotation in annotations:
                    getattr(self, "_plot_" + feature)(axes, annotation, calib=calib, annotation_flag=True, fill=True, alpha=0.2)
        def plot_det(axes):
            for feature in self._configs.visualization.det:
                for detection in detections:
                    getattr(self, "_plot_" + feature)(axes, detection, calib=calib, annotation_flag=False, fill=False)

        if len(annotations) == 0:
            bbox2d = None
        else:
            assert len(annotations) == 1
            bbox2d = expand_bbox(annotations[0].bbox2d, 2.0)

        img = image_tensor.permute(1, 2, 0)

        fig, axes = pyplot.subplots(
            nrows = 3,
            ncols = 1,
            squeeze = False,
            figsize = [10, 25],
            # figsize = [dim / PYPLOT_DPI for dim in image_tensor.shape[2:0:-1]],
        )
        plot_img(axes[0,0], img, 'Both', bbox2d=bbox2d)
        plot_gt(axes[0,0])
        plot_det(axes[0,0])
        plot_img(axes[1,0], img, 'GT', bbox2d=bbox2d)
        plot_gt(axes[1,0])
        plot_img(axes[2,0], img, 'Det', bbox2d=bbox2d)
        plot_det(axes[2,0])


        self._writer.add_figure(mode, fig, index)

    def show_outputs(self, outputs, batch, index):
        gt_maps = batch.gt_map
        cnn_input = normalize(batch.input[0], mean=-TV_MEAN/TV_STD, std=1/TV_STD)

        # CNN input
        fig, axes = pyplot.subplots(figsize=[dim / PYPLOT_DPI for dim in cnn_input.shape[2:0:-1]])
        _ = axes.imshow(cnn_input.permute(1, 2, 0))
        self._writer.add_figure('input', fig, index)

        # CNN outputs
        if self._configs.visualization.cnn_outputs:
            for layer, gt_map in gt_maps.items():
                fig, axes = pyplot.subplots(figsize=[dim / PYPLOT_DPI for dim in gt_map.shape[:1:-1]])
                _ = axes.imshow(gt_map[0, 0].detach().cpu().numpy())
                self._writer.add_figure(layer + '_gt', fig, index)
        # GT Maps
        if self._configs.visualization.cnn_outputs_gtmask:
            for layer, tensor in outputs.items():
                fig, axes = pyplot.subplots(figsize=[dim / PYPLOT_DPI for dim in tensor[0].shape[:1:-1]])
                _ = axes.imshow(tensor[0][0, 0].detach().cpu().numpy())
                self._writer.add_figure(layer + '_output', fig, index)

    def _plot_confidence(self, axes, obj, **kwargs):
        _, _, xmax, ymax = obj.bbox2d
        axes.text(x=xmax, y=ymax,
                  s='conf={0:.2f}'.format(obj.confidence),
                  fontdict={'family': 'monospace',
                            'color':  'white',
                            'size': 'small'},
                  bbox={'color': 'black', 'alpha': 0.5})


    def _plot_bbox2d(self, axes, obj, calib, annotation_flag, **kwargs):
        x1, y1, x2, y2 = obj.bbox2d
        color = self._class_map.get_color(obj.cls)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, **kwargs)
        axes.add_patch(rect)

    def _plot_bbox3d(self, axes, obj, calib, annotation_flag, **kwargs):
        corners_2d = project_3d_pts(
            construct_3d_box(obj.size),
            calib,
            obj.location,
            rot_y=obj.rotation_y,
        )
        coordinates = [corners_2d[:, idx] for idx in BOX_SKELETON]
        color = self._class_map.get_color(obj.cls)
        polygon = patches.Polygon(coordinates, linewidth=2, edgecolor=color, **kwargs)
        axes.add_patch(polygon)

    def _plot_corners(self, axes, obj, **kwargs):
        for corner_xy, color in zip(obj.corners[:, :2], self._corner_colors):
            axes.add_patch(patches.Circle(corner_xy, radius=3, color=color, edgecolor='black'))

    def _plot_keypoints(self, axes, obj, calib, annotation_flag, **kwargs):
        if annotation_flag:
            rotation = matrix_from_yaw(obj.rot_y) if hasattr(obj, 'rot_y') \
                       else obj.rotation
            obj_label = self._class_map.label_from_id(obj.cls) if annotation_flag else obj.cls
            keypoints_3d = self._metadata['objects'][obj_label]['keypoints']
            nbr_kp = keypoints_3d.shape[1]
            keypoints_2d = project_3d_pts(
                keypoints_3d,
                calib,
                obj.location,
                rot_matrix=rotation,
            ).T
            for corner_xy, color in zip(keypoints_2d, self._keypoint_colors):
                axes.add_patch(patches.Circle(corner_xy, radius=3, fill=True, color=color, edgecolor='black'))
        else:
            keypoints_2d = obj.keypoints[:,:2]
            kp_visibility = obj.keypoints[:,2] > 0.5 # Binary signal - make boolean
            for corner_xy, color, kp_visible in zip(keypoints_2d, self._keypoint_colors, kp_visibility):
                if kp_visible:
                    axes.add_patch(patches.Circle(corner_xy, radius=5, fill=False, edgecolor=color))
        # for corner_xy, color in zip(obj.corners.T, self._corner_colors):
        #     axes.add_patch(patches.Circle(corner_xy, radius=3, color=color, edgecolor='black'))

    def _plot_zdepth(self, axes, obj, **kwargs):
        _, ymin, xmax, _ = obj.bbox2d
        axes.text(x=xmax, y=ymin,
                  s='z={0:.2f}m'.format(obj.zdepth),
                  fontdict={'family': 'monospace',
                            'color':  'white',
                            'size': 'small'},
                  bbox={'color': 'black', 'alpha': 0.5})
