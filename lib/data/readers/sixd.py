"""Reading input data in the SIXD common format."""
from os import listdir
from os.path import join
from collections import namedtuple

import numpy as np
from torch import Tensor
from matplotlib.pyplot import cm

from lib.constants import IGNORE_IDX_CLS
from lib.data.loader import Sample
from lib.rigidpose.sixd_toolkit.pysixd.inout import load_cam_params, load_gt
from lib.utils import read_image_to_pt


Annotation = namedtuple('Annotation', ['obj_class', 'bbox2d', 'location', 'rotation'])


class Reader:
    """docstring for Reader."""
    def __init__(self, configs):
        self._configs = configs.data
        self._class_map = ClassMap(configs)
        self._calibration = self._read_calibration()
        self._indices = self._gen_indices()

    def _gen_indices(self):
        indices = []
        train_path = join(self._configs.path, 'train')
        for subdir in listdir(train_path):
            indices.append(len(listdir(join(train_path, subdir, 'rgb'))))
        return indices

    def __len__(self):
        return sum(self._indices)

    def __getitem__(self, index):
        index = int(index)
        data = self._read_data(index)
        annotations = self._read_annotations(index)
        calibration = self._calibration
        return Sample(annotations, data, None, calibration, index)

    def _read_data(self, index):
        dir_ind, img_ind = self._get_indices(index)
        path = join(self._configs.path, 'train', str(dir_ind).zfill(2), 'rgb',
                    str(img_ind).zfill(4) + '.png')
        image = read_image_to_pt(path)
        max_h, max_w = self._configs.img_dims
        return image[:, :max_h, :max_w]

    def _read_annotations(self, index):
        annotations = []
        dir_ind, img_ind = self._get_indices(index)
        gts = load_gt(join(self._configs.path, 'train', str(dir_ind).zfill(2), 'gt.yml'))
        for gt in gts[img_ind]:
            bbox2d = Tensor(gt['obj_bb'])
            bbox2d[2:] += bbox2d[:2]  # x,y,w,h, -> x1,y1,x2,y2
            annotations.append(Annotation(obj_class=self._class_map.id_from_label(gt['obj_id']),
                                          bbox2d=bbox2d,
                                          location=Tensor(gt['cam_t_m2c']),
                                          rotation=Tensor(gt['cam_R_m2c'])))
        return annotations

    def _read_calibration(self):
        path = join(self._configs.path, 'camera.yml')
        intrinsic = load_cam_params(path)['K']
        return np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)

    def _get_indices(self, index):
        dir_ind = np.cumsum(self._indices).searchsorted(index + 1)
        img_ind = index - sum(self._indices[:dir_ind])
        return dir_ind + 1, img_ind


class ClassMap:
    """ClassMap."""
    def __init__(self, configs):
        self._class_labels = listdir(join(configs.data.path, 'train'))

    def id_from_label(self, label):
        try:
            return 2 + list(map(int, self._class_labels)).index(label)
        except ValueError:
            return IGNORE_IDX_CLS

    def label_from_id(self, class_id):
        return self._class_labels[class_id - 2]

    def get_ids(self):
        for label in self._class_labels:
            yield self.id_from_label(label)

    def get_color(self, class_id):
        return cm.Set3(class_id % 12)
