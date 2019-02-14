"""Visualizer."""
from os.path import join
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot, patches

from torchvision.transforms.functional import normalize
from tensorboardX import SummaryWriter

from lib.constants import PYPLOT_DPI, BOX_SKELETON, CORNER_COLORS
from lib.constants import TV_MEAN, TV_STD
from lib.utils import project_3d_pts, construct_3d_box, get_class_map

class Visualizer:
    """Visualizer."""
    def __init__(self, configs):
        self._configs = configs
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

    def save_images(self, batch, output, mode, index, sample=-1):
        calib = batch.calibration[sample]
        image_tensor = normalize(batch.input[sample], mean=-TV_MEAN/TV_STD, std=1/TV_STD)
        frame_id = batch.id[sample]
        annotations = batch.annotation[sample]

        detections = output[frame_id]

        fig, axes = pyplot.subplots(figsize=[dim / PYPLOT_DPI for dim in image_tensor.shape[2:0:-1]])
        axes.axis('off')
        _ = axes.imshow(image_tensor.permute(1, 2, 0))
        for feature in self._configs.visualization.gt:
            for annotation in annotations:
                getattr(self, "_plot_" + feature)(axes, annotation, calib=calib, fill=True, alpha=0.2)
        for feature in self._configs.visualization.det:
            for detection in detections:
                getattr(self, "_plot_" + feature)(axes, detection, calib=calib, fill=False)
        self._writer.add_figure(mode, fig, index)

    def _plot_bbox2d(self, axes, obj, calib, **kwargs):
        x1, y1, x2, y2 = obj.bbox2d
        color = self._class_map.get_color(obj.cls)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, **kwargs)
        axes.add_patch(rect)

    def _plot_bbox3d(self, axes, obj, calib, **kwargs):
        corners_2d = project_3d_pts(
            construct_3d_box(obj.size),
            calib,
            obj.location,
            rot_y=obj.rotation_y,
        )
        coordinates = [corners_2d[:, idx] for idx in BOX_SKELETON]
        color = self._class_map.get_color(obj.cls)
        polygon = patches.Polygon(coordinates, linewidth=2, color=color)
        axes.add_patch(polygon)

    def _plot_corners(self, axes, obj, **kwargs):
        for corner_xy, color in zip(obj.corners.T, self._corner_colors):
            axes.add_patch(patches.Circle(corner_xy, radius=3, color=color, edgecolor='black'))

    def _plot_zdepth(self, axes, obj, **kwargs):
        _, ymin, xmax, _ = obj.bbox2d
        axes.text(x=xmax, y=ymin,
                  s='z={0:.2f}m'.format(obj.zdepth),
                  fontdict={'family': 'monospace',
                            'color':  'white',
                            'size': 'small'},
                  bbox={'color': 'black'})
