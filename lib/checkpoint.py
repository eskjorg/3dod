"""Checkpoint handler."""

import os

from lib.log import Logger
from lib.utils import get_device

class CheckpointHandler:
    """Save and load PyTorch checkpoint."""
    def __init__(self, configs):
        self._configs = configs
        self._logger = Logger(self.__class__.__name__)
        self._best_score = 0
        self._checkpoint_dir = os.path.join(configs.experiment_path, 'checkpoints')
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def init(self, model):
        """Create or load model."""
        load_path = self._configs.checkpoint_load_path
        if load_path:
            self._logger.info('Loading checkpoint from: %s', load_path)
            checkpoint = torch.load(load_path, map_location=get_device())
            model.load_state_dict(checkpoint)
        else:
            model = model.to(get_device())
        return model

    def save(self, model_params, epoch, score):
        if score > self._best_score:
            self._best_score = score
            file_name = 'best_model.pth.tar'
            torch.save(model_params, os.path.join(self._configs.checkpoint_dir, file_name))
        if self._configs.train.backup_epochs:
            file_name = 'epoch{0:03d}.pth.tar'.format(epoch)
            torch.save(model_params, os.path.join(self._configs.checkpoint_dir, file_name))
