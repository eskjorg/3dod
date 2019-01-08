"""Logger."""

import logging

from lib.constants import TRAIN, VAL


class Logger():
    """Logger."""

    def __init__(self, name):
        """Constructor."""
        self._logger = logging.getLogger(name)

    def finish_epoch(self, epoch, mode):
        """Log current epoch."""
        mode = {TRAIN: 'Training', VAL: 'Validation'}[mode]
        self._logger.info('%s epoch %s done!',
                          mode, epoch)

    def log_batch(self, epoch, iteration, loss, task_losses, mode):
        """Log current batch."""
        status_total_loss = ('[{name:s}]  '
                             'Epoch:{epoch:<3d}  '
                             'Iteration:{iteration:<5d}  '.
                             format(name=mode.upper(),
                                    epoch=epoch,
                                    iteration=iteration))
        for statistic, value in loss.items():
            status_total_loss += '{stat:s}:{value:>7.3f}'.format(stat=statistic, value=value)
        self._logger.info(status_total_loss)

        for task_name, task_loss in task_losses.items():
            status_task_loss = '{name:<26s}'.format(name=task_name)
            for statistic, value in task_loss.items():
                status_task_loss += '{stat:s}:{value:>7.3f}'.format(stat=statistic, value=value)
            self._logger.info(status_task_loss)
