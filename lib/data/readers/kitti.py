"""Reading KITTI object data from disk."""
import os
from collections import namedtuple
import numpy as np

from lib.constants import VELODYNE, LABEL_2, CALIB
from lib.utils import read_image_to_pt, read_velodyne_to_pt
from lib.data.loader import Sample


Annotation = namedtuple('Annotation', ['object_class', 'truncation', 'occlusion', 'alpha',
                                       'bounding_box', 'dimension', 'location', 'rotation'])
Calibration = namedtuple('Calibration',
                         ['P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'])


class Reader:
    """docstring for Reader."""
    def __init__(self, configs):
        super(Reader, self).__init__()
        self._configs = configs.data

    def __getitem__(self, index):
        data = self._read_data(index)
        annotations = self._read_annotations(index)
        calibration = self._read_calibration(index)
        return Sample(annotations, data, None, calibration, index)

    def _get_path(self, modality, index):
        root = self._configs.path
        id_str = str(index).zfill(6)
        ext = '.bin' if modality == VELODYNE else '.png'
        return os.path.join(root, modality, id_str + ext)

    def _read_data(self, index):
        data = {}
        for cam_name in self._configs.modalities.cam:
            path = self._get_path(cam_name, index)
            load_type = cam_name.split('_')[-1] is 'color'
            data[cam_name] = read_image_to_pt(path, load_type)
        if VELODYNE in self._configs.modalities.lidar:
            path = self._get_path(VELODYNE, index)
            data[VELODYNE] = read_velodyne_to_pt(path)
        return data

    def _read_annotations(self, index):
        object_class, truncation, occlusion, alpha, bounding_box, dimension, location, rotation \
            = ([] for _ in range(9))

        with open(self._get_path(LABEL_2, index)) as annotations:
            for line in annotations:
                labels = line.split()

                object_class.append(labels[0])
                truncation.append(np.array(labels[1], dtype=np.float32))
                occlusion.append(np.array(labels[2], dtype=np.int32))
                alpha.append(np.array(labels[3], dtype=np.float32))
                bounding_box.append(np.array(labels[4:8], dtype=np.float32))
                dimension.append(np.array(labels[8:11], dtype=np.float32))
                location.append(np.array(labels[11:14], dtype=np.float32))
                rotation.append(np.array(labels[14], dtype=np.float32))

        return Annotation(object_class, truncation, occlusion, alpha, bounding_box, \
            dimension, location, rotation)

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
