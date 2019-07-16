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
from lib.rigidpose.pose_estimator import pflat, deg_cm_error

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

    def save_images(self, batch, cnn_outs, frame_results, mode, index, sample=-1):
        if not any(self._configs.visualization.values()):
            return
        calib = batch.calibration[sample]
        K = calib[:,:3]
        # image_tensor = normalize(batch.input[sample], mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        image_tensor = batch.input[sample] 
        # frame_id = batch.id[sample]
        annotations = batch.annotation[sample]

        anno_lookup = dict(zip([anno.cls for anno in annotations], annotations))

        frame_id = batch.id[sample]
        detections = frame_results[frame_id]

        def assert_single_class_and_get_id():
            assert len(anno_lookup.keys()) == 1
            class_id = list(anno_lookup.keys())[0]
            assert len(detections) <= 1
            if len(detections) == 1:
                assert detections[0].cls == class_id
            return class_id

        def plot_poses(ax, class_ids, annotation, detection):
            for class_id in class_ids:
                class_label = self._class_map.label_from_id(class_id)
                if class_id in anno_lookup:
                    anno = anno_lookup[class_id]
                    self.plot_bbox3d(ax, K, anno.rotation, anno.location.numpy(), *self._metadata['objects'][class_label]['bbox3d'], color='b', linestyle='-', linewidth=1)
            for det in detections:
                if det.cls == class_id and det.pose_results is not None:
                        self.plot_bbox3d(ax, K, det.pose_results['P_est'][:,:3], det.pose_results['P_est'][:,3], *self._metadata['objects'][class_label]['bbox3d'], color='g', linestyle=':', linewidth=1)

        def get_pose_gt_and_est():
            class_id = assert_single_class_and_get_id()
            class_label = self._class_map.label_from_id(class_id)
            anno = anno_lookup[class_id]
            P_gt = np.concatenate([anno.rotation, anno.location.numpy()[:,None]], axis=1)
            if len(detections) > 0:
                det = detections[0]
                P_est = det.pose_results['P_est'] if det.pose_results is not None else None
            else:
                P_est = None
            return P_gt, P_est

        def pose_eval_pretty_print(P_gt, P_est):
            if P_est is None:
                return 'Estimation failed.'
            else:
                deg_error, cm_error = deg_cm_error(P_est[:,:3], P_est[:,[3]], P_gt[:,:3], P_gt[:,[3]], rescale2meter_factor=1e-3)
                return '{:.4} cm, {:.4} deg'.format(cm_error, deg_error)


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
            nrows = 2,
            ncols = 1,
            squeeze = False,
            figsize = [10, 25],
            # figsize = [dim / PYPLOT_DPI for dim in image_tensor.shape[2:0:-1]],
        )

        # Pose
        class_id = assert_single_class_and_get_id()
        P_gt, P_est = get_pose_gt_and_est()
        plot_img(axes[0,0], img, pose_eval_pretty_print(P_gt, P_est), bbox2d=bbox2d)
        # plot_img(axes[0,0], img, 'Pose close-up', bbox2d=bbox2d)
        plot_poses(axes[0,0], [class_id], annotations, detections)

        # Keypoints
        plot_img(axes[1,0], img, 'Both', bbox2d=bbox2d)
        plot_gt(axes[1,0])
        plot_det(axes[1,0])
        # plot_img(axes[1,0], img, 'GT', bbox2d=bbox2d)
        # plot_gt(axes[1,0])
        # plot_img(axes[2,0], img, 'Det', bbox2d=bbox2d)
        # plot_det(axes[2,0])

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
