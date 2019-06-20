"""Load batches for training."""
from collections import namedtuple
from importlib import import_module
import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes
from lib.constants import ANNOTATION, INPUT, GT_MAP, CALIBRATION, ID

from lib.utils import project_3d_pts, construct_3d_box, matrix_from_yaw

Batch = namedtuple('Batch', [ANNOTATION, INPUT, GT_MAP, CALIBRATION, ID, 'targets'])
Sample = namedtuple('Sample', [ANNOTATION, INPUT, GT_MAP, CALIBRATION, ID])

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
        return {
            'dataset': dataset,
            'collate_fn': collate_batch,
            'pin_memory': True,
            'drop_last': True,
            'batch_size': data_configs.batch_size,
            'shuffle': data_configs.shuffle,
            'num_workers': data_configs.num_workers
        }

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
        kpts = torch.from_numpy(get_corners(obj, calib))
        kpts = torch.cat((kpts.float(), torch.ones(1, kpts.shape[1])))
        keypoints.append(kpts.t())

        labels += [obj.cls if obj.cls is not 1 else -100]
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

