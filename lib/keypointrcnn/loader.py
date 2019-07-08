"""Load batches for training."""
from collections import namedtuple
from importlib import import_module
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from nuscenes.nuscenes import NuScenes
from lib.constants import ANNOTATION, INPUT, GT_MAP, CALIBRATION, ID

from lib.utils import project_3d_pts, construct_3d_box, matrix_from_yaw

Batch = namedtuple('Batch', [ANNOTATION, INPUT, GT_MAP, CALIBRATION, ID, 'targets'])
Sample = namedtuple('Sample', [ANNOTATION, INPUT, GT_MAP, CALIBRATION, ID])

class FixedSeededRandomSampler(RandomSampler):
    """
    Tweak RandomSampler to:
        Sample an epoch once, and iterate in this order always.
        Use a random seed for sampling.
    """
    def __init__(self, *args, seed='314159', **kwargs):
        super().__init__(*args, **kwargs)

        # Set random seed
        if seed is not None:
            self._set_seed(seed)

        # Sample an epoch as usual with RandomSampler, but store for re-use
        self._fixed_idx_list = list(super().__iter__())

        # Reset RNG state
        if seed is not None:
            self._reset_rng_state()

    def __iter__(self):
        for idx in self._fixed_idx_list:
            yield idx

    def _set_seed(self, seed):
        self._rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    def _reset_rng_state(self):
        torch.set_rng_state(self._rng_state)

class Loader:
    """docstring for Loader."""
    def __init__(self, modes, configs):
        self._configs = configs
        if configs.data.dataformat == 'nuscenes':
            configs['nusc'] = NuScenes(version='v1.0-trainval', dataroot=configs.data.path, verbose=True)
        self._dataset_module = import_module('lib.data.datasets.%s' % configs.data.dataformat)
        for mode in modes:
            loader_configs = self._get_loader_config(mode)
            loader = DataLoader(**loader_configs)
            setattr(self, mode, loader)

    def _get_loader_config(self, mode):
        dataset = self._dataset_module.get_dataset(self._configs, mode)
        data_configs = getattr(self._configs.loading, mode)
        loader_config = {}
        loader_config['dataset'] = dataset
        loader_config['collate_fn'] = collate_batch
        loader_config['pin_memory'] = True
        loader_config['batch_size'] = data_configs.batch_size
        loader_config['num_workers'] = data_configs.num_workers
        loader_config['drop_last'] = True
        if data_configs.shuffle == True:
            loader_config['sampler'] = RandomSampler(dataset)
        elif data_configs.shuffle == False:
            loader_config['sampler'] = SequentialSampler(dataset)
        elif data_configs.shuffle == 'fixed':
            loader_config['sampler'] = FixedSeededRandomSampler(dataset, seed='314159')
        else:
            # Should not happen
            assert False
        return loader_config

    # def _get_loader_config(self, mode):
    #     dataset = self._dataset_module.get_dataset(self._configs, mode)
    #     data_configs = getattr(self._configs.loading, mode)
    #     return {
    #         'dataset': dataset,
    #         'collate_fn': collate_batch,
    #         'pin_memory': True,
    #         'drop_last': True,
    #         'batch_size': data_configs.batch_size,
    #         'shuffle': data_configs.shuffle,
    #         'num_workers': data_configs.num_workers
    #     }

    def gen_batches(self, mode):
        """Return an iterator over batches."""
        # TODO: This is needed until pytorch pin_memory is fixed. Currently casts namedtuple to list
        # https://github.com/pytorch/pytorch/pull/16440
        for batch in getattr(self, mode):
            batch = Batch(*batch)
            for annotations in batch.annotation:
                for index, annotation in enumerate(annotations):
                    annotations[index] = self._dataset_module.Annotation(*annotation)
            yield batch


def collate_batch(batch_list):
    """Collates for PT data loader."""
    batch_list = [sample for sample in batch_list if sample.annotation]
    annotations, in_data, gt_map, calib, img_id = zip(*batch_list)
    #in_data = torch.stack(in_data)
    gt_map = gt_map[0] and {task: torch.stack([sample[task] for sample in gt_map]) for task in gt_map[0]}
    target = [make_target(img_anno, calib) for img_anno, calib in zip(annotations, calib)]
    return (annotations, in_data, gt_map, calib, img_id, target)

def make_target(annotations, calib):
    # if not annotations:
    #     return {'boxes': torch.Tensor(0, 4),
    #             'labels': torch.Tensor(0),
    #             'keypoints': torch.Tensor(0, 8, 3)}
    keypoints = []
    labels = []
    for obj in annotations:
        kpts = torch.from_numpy(obj.keypoints)
        # kpts = torch.from_numpy(get_corners(obj, calib))
        # kpts = torch.cat((kpts.float(), torch.ones(1, kpts.shape[1])))
        kpts = torch.cat((kpts.float(), torch.from_numpy(obj.kp_visibility[None,:]).float()))
        keypoints.append(kpts.t())

        labels += [obj.cls]
        # labels += [obj.cls if obj.cls is not 1 else -100]
    # return {
    #     'boxes': torch.stack([anno.bbox2d for anno in annotations]),
    #     'labels': torch.Tensor(labels),
    #     'keypoints': torch.stack([torch.zeros((1,1)) for anno in annotations]),
    #     # 'keypoints': torch.stack([anno.keypoints for anno in annotations]),
    # }
    return {'boxes': torch.stack([anno.bbox2d for anno in annotations]),
            'labels': torch.Tensor(labels),
            'keypoints': torch.stack(keypoints)}

def get_corners(obj_annotation, calib):
    if hasattr(obj_annotation, 'corners'):
        return obj_annotation.corners
    else:
        rotation = matrix_from_yaw(obj_annotation.rot_y) if hasattr(obj_annotation, 'rot_y') \
                    else obj_annotation.rotation
        return project_3d_pts(construct_3d_box(obj_annotation.size),
                              calib,
                              obj_annotation.location,
                              rot_matrix=rotation)
