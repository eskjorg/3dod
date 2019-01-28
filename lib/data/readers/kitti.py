"""Reading KITTI object data from disk."""
import os
from collections import namedtuple
import numpy as np
import torch

from lib.constants import VELODYNE, LABEL_2, CALIB, IGNORE_IDX_CLS
from lib.utils import read_image_to_pt, read_velodyne_to_pt
from lib.data.loader import Sample


Annotation = namedtuple('Annotation', ['cls', 'truncation', 'occlusion', 'alpha',
                                       'bbox2d', 'size', 'location', 'rotation'])
Calibration = namedtuple('Calibration',
                         ['P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'])


class Reader:
    """docstring for Reader."""
    def __init__(self, configs):
        self._configs = configs.data
        self._class_map = ClassMap(configs)

    def __len__(self):
        calib_path = os.path.join(self._configs.path, CALIB)
        return len(os.listdir(calib_path))

    def __getitem__(self, index):
        data = self._read_data(index)['image_2']
        annotations = self._read_annotations(index)
        calibration = self._read_calibration(index).P2
        return Sample(annotations, data, None, calibration, index)

    def _get_path(self, modality, index):
        root = self._configs.path
        id_str = str(index).zfill(6)
        extensions = {VELODYNE: '.bin',
                      CALIB: '.txt',
                      LABEL_2: '.txt'}
        ext = extensions.get(modality, '.png')
        return os.path.join(root, modality, id_str + ext)

    def _read_data(self, index):
        data = {}
        for cam_name in self._configs.modalities.cam:
            path = self._get_path(cam_name, index)
            load_type = cam_name.split('_')[-1] in ('2', '3')
            image = read_image_to_pt(path, load_type)
            max_h, max_w = self._configs.img_dims
            data[cam_name] = image[:, :max_h, :max_w]
        if VELODYNE in self._configs.modalities.lidar:
            path = self._get_path(VELODYNE, index)
            data[VELODYNE] = read_velodyne_to_pt(path)
        return data

    def _read_annotations(self, index):
        annotations = []
        with open(self._get_path(LABEL_2, index)) as file:
            for line in file:
                labels = line.split()
                object_class = labels[0]
                labels[1:] = map(float, labels[1:])
                truncation = labels[1]
                occlusion = labels[2]
                rotation = labels[14]
                if rotation == -10 or \
                   truncation > self._configs.threshold.truncation or \
                   occlusion > self._configs.threshold.occlusion:
                    object_class = IGNORE_IDX_CLS
                annotations.append(Annotation(cls=self._class_map.id_from_label(object_class),
                                              truncation=truncation, occlusion=occlusion,
                                              alpha=labels[3],
                                              bbox2d=torch.Tensor(labels[4:8]),
                                              size=torch.Tensor(labels[8:11]),
                                              location=torch.Tensor(labels[11:14]),
                                              rotation=rotation))
        return annotations

    def _read_calibration(self, index):
        params = []
        with open(self._get_path(CALIB, index)) as calibration:
            for line in calibration:
                params.append(np.array(line.split(sep=':')[-1].split(), dtype=np.float32))

        P0 = np.reshape(params[0], (3, 4))
        P1 = np.reshape(params[1], (3, 4))
        P2 = np.reshape(params[2], (3, 4))
        P3 = np.reshape(params[3], (3, 4))

        R0_rect_3x3 = np.reshape(params[4], (3, 3))
        R0_rect_4x4 = np.hstack((np.vstack((R0_rect_3x3, np.array([0., 0., 0.]))),
                                 np.array([[0., 0., 0., 1.]]).T)).astype(np.float32)

        Tr_velo_to_cam = np.vstack((np.reshape(params[5], (3, 4)),
                                    np.array([0., 0., 0., 1.]))).astype(np.float32)

        Tr_imu_to_velo = np.vstack((np.reshape(params[5], (3, 4)),
                                    np.array([0., 0., 0., 1.]))).astype(np.float32)

        return Calibration(P0, P1, P2, P3, R0_rect_4x4, Tr_velo_to_cam, Tr_imu_to_velo)


class ClassMap:
    """ClassMap."""
    def __init__(self, configs):
        self._cls_dict = configs.data.class_map
        self._colors = ['black', 'gray', 'blue', 'red', 'green']

    def id_from_label(self, label):
        return self._cls_dict.get(label, IGNORE_IDX_CLS)

    def label_from_id(self, class_id):
        return next(label for label, id_ in self._cls_dict.items() if id_ is class_id)

    def get_ids(self):
        return set(self._cls_dict.values()) - {IGNORE_IDX_CLS}

    def get_color(self, class_id):
        if isinstance(class_id, str):
            class_id = self.id_from_label(class_id)
        return self._colors[class_id]
