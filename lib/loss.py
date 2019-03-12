"""Loss handler."""
import logging
from collections import defaultdict
import torch
from torch import nn, exp, clamp

from lib.constants import TRAIN, VAL
from lib.constants import IGNORE_IDX_CLS, IGNORE_IDX_REG, IGNORE_IDX_CLSNONMUTEX, PATCH_SIZE
from lib.utils import get_device, get_layers, get_class_map


class LossHandler:
    """LossHandler."""
    def __init__(self, configs, name):
        self._configs = configs
        self._logger = logging.getLogger(name)
        self._class_map = get_class_map(self._configs)

        self._layers = get_layers(self._configs.config_name)
        self._losses = defaultdict(list)

        self._loss_function_dict = self._get_loss_functions(self._layers)

    def _get_loss_functions(self, layers):
        loss_function_dict = {}
        for layer_name, layer_spec in self._layers.items():
            if layer_spec['loss'] == 'CE':
                loss_function_dict[layer_name] = self._get_ce_loss(layer_name)
            elif layer_spec['loss'] == 'BCE':
                loss_function_dict[layer_name] = self._get_bce_loss(layer_name)
            elif layer_spec['loss'] == 'L1':
                loss_function_dict[layer_name] = nn.L1Loss(reduction='none').to(get_device())
            else:
                raise NotImplementedError("{} loss not implemented.".format(layer_spec['loss']))
        return loss_function_dict

    def _get_ce_loss(self, layer_name):
        if layer_name == 'clsgroup':
            group_ids = self._class_map.get_group_ids()
            weight = torch.zeros((len(group_ids)+2,))
            weight[0] = 1.0
            weight[group_ids] = self._layers[layer_name]['fg_upweight_factor']
            weight /= torch.sum(weight[[0]+group_ids])
        else:
            weight = None
        return nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_IDX_CLS).to(get_device())

    def _get_bce_loss(self, layer_name):
        assert IGNORE_IDX_CLS == 1
        assert layer_name == 'clsnonmutex'

        if self._layers['clsnonmutex']['ignore_bg']:
            # No considerable class imbalance if background is ignored
            pos_weight = None
        else:
            nbr_classes = len(self._class_map.get_ids())
            # weight = torch.ones((nbr_classes,))
            img_height, img_width = self._configs.data.img_dims
            IMBALANCE = img_height*img_width/float(PATCH_SIZE**2) # Background is more common
            # IMBALANCE = 10000.0 # Background is more common
            pos_weight = IMBALANCE * torch.ones((nbr_classes, *self._configs.target_dims))
        return nn.BCEWithLogitsLoss(
            # weight = weight,
            pos_weight = pos_weight,
            reduction = 'none',
        ).to(get_device())

    def calc_loss(self, gt_maps, outputs_cnn):
        def calc_task_loss(layer_name, layer_outputs, gt_map, lossweight_map=None):
            tensor, ln_var = layer_outputs
            if self._layers[layer_name]['loss'] == 'CE':
                assert lossweight_map is None
                task_loss = self._loss_function_dict[layer_name](tensor, gt_map[:, 0])
            elif self._layers[layer_name]['loss'] == 'BCE':
                assert layer_name == 'clsnonmutex'
                assert lossweight_map is None
                task_loss = self._loss_function_dict[layer_name](tensor, gt_map)
                mask_loss_applied = gt_map.ne(IGNORE_IDX_CLSNONMUTEX)
                # mask_loss_applied = torch.abs(gt_map - IGNORE_IDX_CLSNONMUTEX) > 1e-6
                task_loss = task_loss * mask_loss_applied.float()
                # clamp below is a trick to avoid 0 / 0 = NaN, and instead perform 0 / 1 = 0. Works because denominator will be either 0 or >= 1 (sum of boolean -> non-negative int).
                task_loss = task_loss.sum() / clamp(mask_loss_applied.sum(), min=1)
            elif self._layers[layer_name]['loss'] == 'L1':
                PROJECTION = False
                if PROJECTION:
                    # Permute axes for easier mask indexing
                    pred_perm = tensor.permute(1,0,2,3)
                    gt_perm = gt_map.permute(1,0,2,3)
                    ln_var_perm = ln_var.permute(1,0,2,3)

                    # Determine which predictions are "long" enough to be able to determine their direction
                    pred_perm_norm = torch.norm(pred_perm, dim=0)
                    eps = 1e-6
                    has_dir = pred_perm_norm >= eps
                    # no_dir = has_dir ^ 1
                    no_dir = pred_perm_norm < eps

                    task_loss_perm = torch.empty_like(pred_perm, dtype=torch.float32)

                    def rotate_tensor_90(t1):
                        t2 = t1.flip(0)
                        t2[0,:] = -t2[0,:]
                        return t2

                    # If norm < eps, projection is avoided, since direction is undefined.
                    # Instead, average ln_var is applied in both x & y direction.
                    task_loss_perm[:,no_dir] = self._loss_function_dict[layer_name](pred_perm[:,no_dir], gt_perm[:,no_dir])
                    avg_ln_var = ln_var_perm[:,no_dir].mean(dim=0)
                    task_loss_perm[:,no_dir] = task_loss_perm[:,no_dir] * exp(-avg_ln_var) + avg_ln_var

                    # Determine new basis from direction of predictions
                    axis1 = pred_perm[:,has_dir] / pred_perm_norm[has_dir].unsqueeze(0)
                    axis2 = rotate_tensor_90(axis1)

                    # Carry out projection to determine new coefficients
                    pred1 = pred_perm_norm[has_dir]
                    pred2 = torch.sum(pred_perm[:,has_dir]*axis2, dim=0)
                    gt1 = torch.sum(gt_perm[:,has_dir]*axis1, dim=0)
                    gt2 = torch.sum(gt_perm[:,has_dir]*axis2, dim=0)
                    task_loss_perm[0,has_dir] = self._loss_function_dict[layer_name](pred1, gt1) * exp(-ln_var_perm[0,has_dir]) + ln_var_perm[0,has_dir]
                    task_loss_perm[1,has_dir] = self._loss_function_dict[layer_name](pred2, gt2) * exp(-ln_var_perm[1,has_dir]) + ln_var_perm[1,has_dir]

                    # Permute back
                    task_loss = task_loss_perm.permute(1,0,2,3)

                else:
                    task_loss = self._loss_function_dict[layer_name](tensor, gt_map)
                    task_loss = task_loss * exp(-ln_var) + ln_var

                task_loss = task_loss * gt_map.ne(IGNORE_IDX_REG).float()
                if lossweight_map is not None:
                    task_loss = task_loss * torch.unsqueeze(lossweight_map, 1)
                # clamp below is a trick to avoid 0 / 0 = NaN, and instead perform 0 / 1 = 0. Works because denominator will be either 0 or >= 1 (sum of boolean -> non-negative int).
                task_loss = task_loss.sum() / clamp(gt_map.ne(IGNORE_IDX_REG).sum(), min=1)
            else:
                raise NotImplementedError("{} loss not implemented.".format(self._layers[layer_name]['loss']))
            return task_loss
        loss = 0
        for layer_name in self._layers:
            if self._layers[layer_name]['cls_specific_heads']:
                # Separate GT map for every class
                nbr_classes = len(self._class_map.get_ids())

                REPORT_LOSS_PER_CLASS = False

                if layer_name == 'keypoint':
                    lossweight_maps = gt_maps['clsnonmutex'].to(get_device(), non_blocking=True)

                if not REPORT_LOSS_PER_CLASS:
                    task_loss_avg = 0
                for cls_id in self._class_map.get_ids():
                    class_label = self._class_map.label_from_id(cls_id)
                    head_name = '{}_{}'.format(layer_name, class_label)
                    gt_map = gt_maps[head_name].to(get_device(), non_blocking=True)
                    lossweight_map = lossweight_maps[:,cls_id-2,:,:] if layer_name == 'keypoint' else None
                    task_loss = calc_task_loss(layer_name, outputs_cnn[head_name], gt_map, lossweight_map=lossweight_map)
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
                task_loss = calc_task_loss(layer_name, outputs_cnn[layer_name], gt_map)
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
            status_total_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
                                                                     value=sum(value.values()))
        self._logger.info(status_total_loss)

        for task_name in self._losses.keys():
            status_task_loss = '{name:<26s}'.format(name=task_name)
            for statistic, value in losses.items():
                status_task_loss += '{stat:s}: {value:>7.7f}   '.format(stat=statistic,
                                                                        value=value[task_name])
            self._logger.info(status_task_loss)

    def finish_epoch(self, epoch, mode):
        """Log current epoch."""
        mode = {TRAIN: 'Training', VAL: 'Validation'}[mode]
        self._logger.info('%s epoch %s done!',
                          mode, epoch)
        self._losses = defaultdict(list)
