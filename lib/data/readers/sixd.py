"""Reading input data in the SIXD common format."""
from os.path import join
from collections import namedtuple
import yaml

import numpy as np
from torch import Tensor
from matplotlib.pyplot import cm

from lib.constants import IGNORE_IDX_CLS
from lib.data.loader import Sample
from lib.utils import read_image_to_pt
from lib.utils import listdir_nohidden


def get_metadata(configs):
    path = join(configs.data.path, 'models', 'models_info.yml')
    with open(path, 'r') as file:
        models_info = yaml.load(file)
    def build_kp_array(obj_anno):
        return np.array([
            obj_anno['kp_x'],
            obj_anno['kp_y'],
            obj_anno['kp_z'],
        ])
    return {
        'objects': {obj_label: {
            # NOTE: ClassMap.id_from_label could be called when needed instead of storing ids. Unless performance issue..?
            # 'obj_id': ClassMap.id_from_label(obj_label),
            'keypoints': build_kp_array(obj_anno),
        } for obj_label, obj_anno in models_info.items()},
    }

Annotation = namedtuple('Annotation', ['cls', 'bbox2d', 'size', 'location', 'rotation'])


class Reader:
    """docstring for Reader."""
    def __init__(self, configs):
        self._configs = configs.data
        self._class_map = ClassMap(configs)
        self._n_class_instances = self._init_class_instances()
        self._models = self._init_models()

    def _init_class_instances(self):
        indices = []
        train_path = join(self._configs.path, self._configs.subdir)
        for subdir in listdir_nohidden(train_path):
            indices.append(len(listdir_nohidden(join(train_path, subdir, 'rgb'))))
        return indices

    def _init_models(self):
        path = join(self._configs.path, 'models', 'models_info.yml')
        with open(path, 'r') as file:
            return yaml.load(file)

    def __len__(self):
        return sum(self._n_class_instances)

    def __getitem__(self, index):
        index = int(index)
        dir_ind, img_ind = self._get_indices(index)
        dir_path = join(self._configs.path, self._configs.subdir, self._class_map[dir_ind])
        data = self._read_data(dir_path, img_ind)
        annotations = self._read_annotations(dir_path, img_ind, self._models[dir_ind + 1])
        calibration = self._read_calibration(dir_path, img_ind)
        return Sample(annotations, data, None, calibration, index)

    def _read_data(self, dir_path, img_ind):
        path = join(dir_path, 'rgb', str(img_ind).zfill(4) + '.png')
        image = read_image_to_pt(path)
        max_h, max_w = self._configs.img_dims
        return image[:, :max_h, :max_w]

    def _read_annotations(self, dir_path, img_ind, model):
        annotations = []
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
        return np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)

    def _get_indices(self, index):
        dir_ind = np.cumsum(self._n_class_instances).searchsorted(index + 1)
        img_ind = index - sum(self._n_class_instances[:dir_ind])
        return dir_ind, img_ind


class ClassMap:
    """ClassMap."""
    def __init__(self, configs):
        self.class_labels = sorted(listdir_nohidden(join(configs.data.path, configs.data.subdir)))

    def __getitem__(self, idx):
        return self.class_labels[idx]

    def id_from_label(self, label):
        """In network, 0 and 1 are reserved for background and don't_care"""
        return 1 + label

    def label_from_id(self, class_id):
        return self.class_labels[class_id - 2]

    def get_ids(self):
        return range(2, 2 + len(self.class_labels))

    def get_color(self, class_id):
        if isinstance(class_id, str):
            class_id = self.id_from_label(class_id)
        return cm.Set3(class_id % 12)
