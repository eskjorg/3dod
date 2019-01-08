"""Checkpoint handler."""

import os

from log import Logger
from util import get_device

class CheckpointHandler:
    """Save and load PyTorch checkpoint."""
    def __init__(self, settings):
        self._settings = settings
        self._logger = Logger(self.__class__.__name__)
        self._best_score = 0
        os.makedirs(settings.checkpoint_dir, exist_ok=True)

    def init(self, model):
        """Create or load model."""
        if self._settings.load_path:
            self._logger.info('Loading checkpoint from: %s', self._settings.load_path)
            checkpoint = torch.load(self._settings.load_path, map_location=get_device())
            model.load_state_dict(checkpoint)
        else:
            model = model.to(get_device())

    def save(self, model_params, epoch, score):
        if score > self._best_score:
            self._best_score = score
            file_name = 'best_model.pth.tar'
            torch.save(model_params, os.path.join(self._settings.checkpoint_dir, file_name))
        if self._settings.train.backup_epochs:
            file_name = 'epoch{0:03d}.pth.tar'.format(epoch)
            torch.save(model_params, os.path.join(self._settings.checkpoint_dir, file_name))
