"""Visualizer."""
import tensorboardX


class Visualizer:
    """Visualizer."""
    def __init__(self):
        super(Visualizer, self).__init__()

    def save_images(self, epoch, batch, detections, mode):
        print('Visualizer.save_images() not yet implemented')

    def report_loss(self, epoch, losses, mode):
        print('Visualizer.report_loss() not yet implemented')

    def report_score(self, epoch, score, mode):
        print('Visualizer.report_score() not yet implemented')
