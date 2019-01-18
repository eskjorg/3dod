"""Visualizer."""
from os.path import join
from tensorboardX import SummaryWriter


class Visualizer:
    """Visualizer."""
    def __init__(self, configs):
        self._configs = configs
        self._writer = SummaryWriter(join(configs.experiment_path, 'visual'))

    def save_images(self, epoch, batch, detections, mode):
        #writer.add_image('imresult', x, iteration)
        print("Not yet implemented: ", 'Visualizer.save_images()')

    def report_loss(self, epoch, losses, mode):
        self._writer.add_scalar('loss/{}'.format(mode), sum(losses.values()), epoch)
        self._writer.add_scalars('task_losses/{}'.format(mode), losses, epoch)

    def report_score(self, epoch, score, mode):
        self._writer.add_scalar('score/{}'.format(mode), score, epoch)
