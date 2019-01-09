"""Load batches for training."""
import os
import torch.utils.data as ptdata

from lib.constants import SETTINGS_PATH
from lib.data import process

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
        return {
            'dataset': dataset,
            'collate_fn': process.collate_batch,
            'drop_last': True,
            'batch_size': self._configs.batch_size,
            'shuffle': self._configs.shuffle,
            'num_workers': self._configs.shuffle
        }

    def _get_indices(self, mode):
        """Load indices for train/val/test split.

        Returns:
            list: Indices

        """
        path = os.path.join(SETTINGS_PATH,
                            self._configs.name,
                            self._configs.split_dir,
                            '{}.txt'.format(mode))
        with open(path) as file:
            return file.read().splitlines()

    def gen_batches(self, mode):
        """Return an iterator over batches."""
        return getattr(self, mode)


class Dataset(ptdata.Dataset):
    """docstring for Dataset."""
    def __init__(self, configs, mode):
        self._configs = configs
        self._reader = None  # TODO:
        self._processor = None  # TODO:
        super(Dataset, self).__init__()

    def __len__(self):
        return(len(self._reader))

    def __getitem__(self, idx):
        raw_sample = self._reader[idx]
        sample = self._processor.get_sample(raw_sample)
        return sample
