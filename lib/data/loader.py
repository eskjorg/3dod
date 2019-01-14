"""Load batches for training."""
import os
from collections import namedtuple
from importlib import import_module
import torch.utils.data as ptdata

import numpy as np

from lib.constants import SETTINGS_PATH, TRAIN, VAL
from lib.constants import ANNOTATION, INPUT, GT, CALIBRATION, ID
from lib.data.gt import GTGenerator


Sample = namedtuple('Sample', [ANNOTATION, INPUT, GT, CALIBRATION, ID])
Batch = namedtuple('Batch', [ANNOTATION, INPUT, GT, CALIBRATION, ID])


class Loader:
    """docstring for Loader."""
    def __init__(self, modes, configs):
        self._configs = configs
        for mode in modes:
            loader_configs = self._get_loader_config(mode)
            loader = ptdata.DataLoader(**loader_configs)
            setattr(self, mode, loader)

    def _get_loader_config(self, mode):
        dataset = ptdata.Subset(dataset=Dataset(self._configs, mode),
                                indices=self._get_indices(mode))
        data_configs = getattr(self._configs.loading, mode)
        return {
            'dataset': dataset,
            'collate_fn': collate_batch,
            'drop_last': True,
            'batch_size': data_configs.batch_size,
            'shuffle': data_configs.shuffle,
            'num_workers': data_configs.num_workers
        }

    def _get_indices(self, mode):
        """Load indices for train/val/test split.

        Returns:
            list: Indices

        """
        path = os.path.join(SETTINGS_PATH,
                            self._configs.config_load_path,
                            self._configs.data.split_dir,
                            '{}.txt'.format(mode))
        with open(path) as file:
            return file.read().splitlines()

    def gen_batches(self, mode):
        """Return an iterator over batches."""
        return getattr(self, mode)


def collate_batch(batch_list):
    """Collates for PT data loader."""
    annotations, in_data, gt_mask, calib, img_id = zip(*batch_list)
    in_data = np.stack(in_data)
    gt_mask = {task: np.stack([sample[task] for sample in gt_mask]) for task in gt_mask[0]}
    calib = map(lambda x: x.P0, gt_mask)
    return Batch(annotations, in_data, gt_mask, calib, img_id)


class Dataset(ptdata.Dataset):
    """docstring for Dataset."""
    def __init__(self, configs, mode):
        self._configs = configs
        self._mode = mode
        self._reader = self._get_reader()
        self._augmenter = None  # TODO: or not ?
        #elf._processor = self._get_processor()
        self._gt_generator = GTGenerator(self._configs)
        super(Dataset, self).__init__()

    def __len__(self):
        return(len(self._reader))

    def __getitem__(self, idx):
        annotations, data, gt, calibration, index = self._reader[idx]
        if self._mode in (TRAIN, VAL):
            gt = self._gt_generator.generate(annotations, calibration)
        if self._mode is TRAIN:
            #data = self._augmenter.augment(data)
            None
        #sample = self._processor.get_sample(raw_sample)
        return Sample(annotations, data, gt, calibration, index)

    def _get_reader(self):
        reader_module = import_module('lib.data.readers.{}'.format(self._configs.data.dataset))
        return reader_module.Reader(self._configs)

    #def _get_processor(self):
    #    return getattr(process, self._mode.capitalize())()
