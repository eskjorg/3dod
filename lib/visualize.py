"""Visualizer."""
import tensorboardX


class Visualizer:
    """Visualizer."""
    def __init__(self):
        super(Visualizer, self).__init__()

    def save_images(self, epoch, batch, detections, mode):
        print("Not yet implemented: ", 'Visualizer.save_images()')

    def report_loss(self, epoch, losses, mode):
        print("Not yet implemented: ", 'Visualizer.report_loss()')

    def report_score(self, epoch, score, mode):
        print("Not yet implemented: ", 'Visualizer.report_score()')
