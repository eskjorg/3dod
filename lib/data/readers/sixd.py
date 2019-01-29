"""Reading input data in the SIXD common format."""
from os import listdir
from os.path import join
from collections import namedtuple
import yaml

import numpy as np
from torch import Tensor
from matplotlib.pyplot import cm

from lib.constants import IGNORE_IDX_CLS
from lib.data.loader import Sample
from lib.utils import read_image_to_pt


Annotation = namedtuple('Annotation', ['cls', 'bbox2d', 'size', 'location', 'rotation'])


class Reader:
    """docstring for Reader."""
    def __init__(self, configs):
        self._configs = configs.data
        self._class_map = ClassMap(configs)
        self._indices = self._init_indices()
        self._models = self._init_models()

    def _init_indices(self):
        indices = []
        train_path = join(self._configs.path, 'train')
        for subdir in listdir(train_path):
            indices.append(len(listdir(join(train_path, subdir, 'rgb'))))
        return indices

    def _init_models(self):
        path = join(self._configs.path, 'models', 'models_info.yml')
        with open(path, 'r') as file:
            return yaml.load(file)

    def __len__(self):
        return sum(self._indices)

    def __getitem__(self, index):
        index = int(index)
        dir_ind, img_ind = self._get_indices(index)
        dir_path = join(self._configs.path, 'train', str(dir_ind).zfill(2))
        data = self._read_data(dir_path, img_ind)
        annotations = self._read_annotations(dir_path, img_ind, dir_ind)
        calibration = self._read_calibration(dir_path, img_ind)
        return Sample(annotations, data, None, calibration, index)

    def _read_data(self, dir_path, img_ind):
        path = join(dir_path, 'rgb', str(img_ind).zfill(4) + '.png')
        image = read_image_to_pt(path)
        max_h, max_w = self._configs.img_dims
        return image[:, :max_h, :max_w]

    def _read_annotations(self, dir_path, img_ind, dir_ind):
        annotations = []
        model = self._models[dir_ind]
        size = (model['size_x'], model['size_y'], model['size_z'])

        with open(join(dir_path, 'gt.yml'), 'r') as file:
            gts = yaml.load(file, Loader=yaml.CLoader)
        for gt in gts[img_ind]:
            bbox2d = Tensor(gt['obj_bb'])
            bbox2d[2:] += bbox2d[:2]  # x,y,w,h, -> x1,y1,x2,y2
            annotations.append(Annotation(cls=self._class_map.id_from_label(gt['obj_id']),
                                          bbox2d=bbox2d,
                                          size=Tensor(size),
                                          location=Tensor(gt['cam_t_m2c']),
                                          rotation=np.array(gt['cam_R_m2c']).reshape((3, 3))))
        return annotations

    def _read_calibration(self, dir_path, img_ind):
        with open(join(dir_path, 'info.yml'), 'r') as file:
            obj_info = yaml.load(file, Loader=yaml.CLoader)[img_ind]
        intrinsic = np.reshape(obj_info['cam_K'], (3, 3))
        intrinsic[2, 2] *= 1000  # Using mm scale
        return np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)

    def _get_indices(self, index):
        dir_ind = np.cumsum(self._indices).searchsorted(index + 1)
        img_ind = index - sum(self._indices[:dir_ind])
        return dir_ind + 1, img_ind


class ClassMap:
    """ClassMap."""
    def __init__(self, configs):
        self._class_labels = sorted(listdir(join(configs.data.path, 'train')))

    def id_from_label(self, label):
        try:
            return 2 + list(map(int, self._class_labels)).index(int(label))
        except ValueError:
            return IGNORE_IDX_CLS

    def label_from_id(self, class_id):
        return self._class_labels[class_id - 2]

    def get_ids(self):
        for label in self._class_labels:
            yield self.id_from_label(label)

    def get_color(self, class_id):
        if isinstance(class_id, str):
            class_id = self.id_from_label(class_id)
        return cm.Set3(class_id % 12)
