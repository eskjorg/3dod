"""Visualizer."""
from os.path import join
import shutil

from matplotlib import pyplot, patches

from tensorboardX import SummaryWriter

from lib.constants import PYPLOT_DPI

class Visualizer:
    """Visualizer."""
    def __init__(self, configs):
        self._configs = configs
        vis_path = join(configs.experiment_path, 'visual')
        shutil.rmtree(vis_path, ignore_errors=True)
        self._writer = SummaryWriter(vis_path)
        self._color_obj = ['black', 'gray', 'blue', 'red', 'green']
        self._color_corner = ['magenta', 'cyan', 'yellow', 'green',
                              'lime', 'blue', 'purple', 'orange']

    def report_loss(self, epoch, losses, mode):
        self._writer.add_scalar('loss/{}'.format(mode), sum(losses.values()), epoch)
        self._writer.add_scalars('task_losses/{}'.format(mode), losses, epoch)

    def report_score(self, epoch, score, mode):
        self._writer.add_scalar('score/{}'.format(mode), score, epoch)

    def save_images(self, epoch, batch, output, mode, sample=-1):
        calib = batch.calibration[sample]
        image_tensor = batch.input[sample]
        frame_id = batch.id[sample]
        annotations = batch.annotation[sample]

        detections = output[frame_id]

        fig, axes = pyplot.subplots(figsize=[dim / PYPLOT_DPI for dim in image_tensor.shape[2:0:-1]])
        axes.axis('off')
        _ = axes.imshow(image_tensor.permute(1, 2, 0))
        for feature in self._configs.visualization.gt:
            getattr(self, "_plot_" + feature)(axes, annotations, calib=calib, is_gt=True)
        for feature in self._configs.visualization.det:
            getattr(self, "_plot_" + feature)(axes, detections, calib=calib, is_gt=False)
        self._writer.add_figure(mode, fig, epoch)

    def _plot_bbox2d(self, axes, objects, **kwargs):
        is_gt = kwargs['is_gt']
        for obj in objects:
            class_id = obj[0] if is_gt else obj['class']
            color = self._color_obj[class_id]
            alpha = 0.2 if is_gt else 1.0
            x1, y1, x2, y2 = obj[4] if is_gt else obj['bbox2d']
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor=color, linewidth=2, alpha=alpha, fill=is_gt)
            axes.add_patch(rect)

    def _plot_corners(self, axes, objects, **kwargs):
        for obj in objects:
            for corner_xy, color in zip(obj['corners'].T, self._color_corner):
                axes.add_patch(patches.Circle(corner_xy, color=color))

    def _plot_zdepth(self, axes, objects, **kwargs):
        for obj in objects:
            x, y, _, _ = obj['bbox2d']
            text = 'zdepth={0:.2f}m'.format(obj['zdepth'])
            self._plot_text(box[0], box[1], text, 'white', color, self._zorder + 1)

            axes.text(x, y, s)
