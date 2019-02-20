"""Reading nuscenes object data from disk."""
from collections import namedtuple

import numpy as np
from matplotlib.pyplot import cm
import torch

from nuscenes.utils.geometry_utils import BoxVisibility, view_points

from lib.constants import TRAIN, VAL, IGNORE_IDX_CLS
from lib.data.loader import Sample
from lib.data.maps import GtMapsGenerator
from lib.utils import read_image_to_pt


Annotation = namedtuple('Annotation', ['cls', 'bbox2d', 'size', 'location', 'rotation', 'corners'])


def get_metadata(configs):
    return {}


def get_dataset(configs, mode):
    return NuscenesDataset(configs, mode)


class NuscenesDataset(torch.utils.data.Dataset):
    def __init__(self, configs, mode):
        self._configs = configs.data
        self._mode = mode
        self._nusc = configs.nusc
        self._data_tokens = self._init_data_tokens()
        self._class_map = ClassMap(configs)
        self._gt_map_generator = GtMapsGenerator(configs)

    def _init_data_tokens(self):
        tokens = []
        scenes = self._configs.scenes[self._mode]
        channels = self._configs.channels
        def is_keyframe(sample_data, sample):
            channel = sample_data['channel']
            return sample['data'][channel] == sample_data['token']
        for sample_data in self._nusc.sample_data:
            sample = self._nusc.get('sample', sample_data['sample_token'])
            if not self._configs.keyframes_only or is_keyframe(sample_data, sample):
                scene = self._nusc.get('scene', sample['scene_token'])['name']
                channel = sample_data['channel']
                if scene in scenes and channel in channels:
                    tokens.append(sample_data['token'])
        return tokens

    def __len__(self):
        return len(self._data_tokens)

    def __getitem__(self, index):
        data_token = self._data_tokens[index]
        path, boxes, calib = self._nusc.get_sample_data(data_token, box_vis_level=BoxVisibility.ANY)

        data = read_image_to_pt(path)
        max_h, max_w = self._configs.img_dims
        data = data[:, :max_h, :max_w]

        to_kitti_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        annotations = []
        for box in boxes:
            corners = view_points(box.corners(), calib, normalize=True)[:2]
            annotations.append(Annotation(cls=self._class_map.id_from_label(box.name),
                                          bbox2d=self._get_bbox2d(corners),
                                          size=box.wlh[[2, 0, 1]],  # WLH -> HWL
                                          location=box.center,
                                          rotation=box.orientation.rotation_matrix @ to_kitti_rot,
                                          corners=corners))

        calibration = np.concatenate((calib, np.zeros((3, 1))), axis=1)

        gt_maps = self._mode in (TRAIN, VAL) and \
                  self._gt_map_generator.generate(annotations, calibration)

        return Sample(annotations, data, gt_maps, calibration, id=data_token)

    def _get_bbox2d(self, corners):
        xmin = max(min(corners[0, :]), 0)
        ymin = max(min(corners[1, :]), 0)
        xmax = min(max(corners[0, :]), self._configs.img_dims[1])
        ymax = min(max(corners[1, :]), self._configs.img_dims[0])
        return torch.Tensor((xmin, ymin, xmax, ymax))


class ClassMap:
    """ClassMap."""
    def __init__(self, configs):
        self._cls_dict = configs.data.class_map

    def id_from_label(self, label):
        return self._cls_dict.get(label, IGNORE_IDX_CLS)

    def label_from_id(self, class_id):
        return next(label for label, id_ in self._cls_dict.items() if id_ is class_id)

    def get_ids(self):
        return set(self._cls_dict.values()) - {IGNORE_IDX_CLS}

    def get_color(self, class_id):
        if isinstance(class_id, str):
            class_id = self.id_from_label(class_id)
        return cm.Set3(class_id % 12)
