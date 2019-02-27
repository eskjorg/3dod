"""Loss handler."""
import logging
from collections import defaultdict
from torch import nn, exp, clamp

from lib.constants import TRAIN, VAL
from lib.constants import LN_2, IGNORE_IDX_CLS, IGNORE_IDX_REG
from lib.utils import get_device, get_layers, get_class_map


class LossHandler:
    """LossHandler."""
    def __init__(self, configs, name):
        self._configs = configs
        self._logger = logging.getLogger(name)
        self._class_map = get_class_map(self._configs)

        self._layers = get_layers(self._configs.config_name)
        self._losses = defaultdict(list)

        self._ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX_CLS).to(get_device())
        self._l1_loss = nn.L1Loss(reduction='none').to(get_device())

    def calc_loss(self, gt_maps, outputs_cnn):
        def calc_task_loss(layer_name, tensor, gt_map):
            if self._layers[layer_name]['loss'] == 'CE':
                task_loss = self._ce_loss(tensor, gt_map[:, 0])
            elif self._layers[layer_name]['loss'] == 'L1':
                task_loss = self._l1_loss(tensor, gt_map)
                if outputs_ln_b:
                    ln_b = outputs_ln_b[layer_name]
                    task_loss = task_loss * exp(-ln_b) + LN_2 + ln_b
                task_loss = task_loss * gt_map.ne(IGNORE_IDX_REG).float()
                # clamp below is a trick to avoid 0 / 0 = NaN, and instead perform 0 / 1 = 0. Works because denominator will be either 0 or >= 1 (sum of boolean -> non-negative int).
                task_loss = task_loss.sum() / clamp(gt_map.ne(IGNORE_IDX_REG).sum(), min=1)
            else:
                raise NotImplementedError("{} loss not implemented.".format(self._layers[layer_name]['loss']))
            return task_loss
        loss = 0
        outputs_task, outputs_ln_b = outputs_cnn
        for layer_name, tensor in outputs_task.items():
            if self._layers[layer_name]['cls_specific_heads']:
                # Separate GT map for every class
                nbr_classes = len(self._class_map.get_ids())

                REPORT_LOSS_PER_CLASS = False

                if not REPORT_LOSS_PER_CLASS:
                    task_loss_avg = 0
                for cls_id in self._class_map.get_ids():
                    class_label = self._class_map.label_from_id(cls_id)
                    head_name = '{}_{}'.format(layer_name, class_label)
                    gt_map = gt_maps[head_name].to(get_device(), non_blocking=True)
                    task_loss = calc_task_loss(layer_name, tensor, gt_map)
                    loss += task_loss * self._layers[layer_name]['loss_weight'] / float(nbr_classes)
                    if REPORT_LOSS_PER_CLASS:
                        self._losses[head_name].append(task_loss.item())
                    else:
                        task_loss_avg += task_loss / float(nbr_classes)
                if not REPORT_LOSS_PER_CLASS:
                    self._losses[layer_name].append(task_loss_avg.item())
            else:
                # Single GT map - shared among all classes
                gt_map = gt_maps[layer_name].to(get_device(), non_blocking=True)
                task_loss = calc_task_loss(layer_name, tensor, gt_map)
                loss += task_loss * self._layers[layer_name]['loss_weight']
                self._losses[layer_name].append(task_loss.item())
        return loss

    def get_averages(self, num_batches=0):
        avg_losses = defaultdict(int)
        for loss_name, loss_list in self._losses.items():
            latest_losses = loss_list[-num_batches:]
            avg_losses[loss_name] = sum(latest_losses) / len(latest_losses)
        return avg_losses

    def log_batch(self, epoch, iteration, mode):
        """Log current batch."""
        losses = {
            'Loss': self.get_averages(num_batches=1),
            'Moving Avg': self.get_averages(num_batches=self._configs.logging.avg_window_size),
            'Average': self.get_averages(num_batches=0)
        }
        status_total_loss = ('[{name:s}]  '
                             'Epoch:{epoch:<3d}  '
                             'Iteration:{iteration:<5d}  '.
                             format(name=mode.upper(),
                                    epoch=epoch,
                                    iteration=iteration))
        for statistic, value in losses.items():
            status_total_loss += '{stat:s}: {value:>7.3f}   '.format(stat=statistic,
                                                                     value=sum(value.values()))
        self._logger.info(status_total_loss)

        for task_name in self._losses.keys():
            status_task_loss = '{name:<26s}'.format(name=task_name)
            for statistic, value in losses.items():
                status_task_loss += '{stat:s}: {value:>7.3f}   '.format(stat=statistic,
                                                                        value=value[task_name])
            self._logger.info(status_task_loss)

    def finish_epoch(self, epoch, mode):
        """Log current epoch."""
        mode = {TRAIN: 'Training', VAL: 'Validation'}[mode]
        self._logger.info('%s epoch %s done!',
                          mode, epoch)
        self._task_losses = defaultdict(list)
