"""Reading input data in the SIXD common format."""
from os.path import join
from collections import namedtuple, OrderedDict
import yaml

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from matplotlib.pyplot import cm

from lib.constants import IGNORE_IDX_CLS, TRAIN, VAL, NBR_KEYPOINTS, PATCH_SIZE
from lib.data.loader import Sample
from lib.data.maps import GtMapsGenerator
from lib.utils import read_image_to_pt
from lib.utils import listdir_nohidden
from lib.utils import get_metadata
from lib.utils import project_3d_pts


def get_metadata(configs):
    path = join(configs.data.path, 'models', 'models_info.yml')
    with open(path, 'r') as file:
        models_info = yaml.load(file, Loader=yaml.CLoader)
    def rows2array(obj_anno, prefix):
        return np.array([
            obj_anno[prefix + '_x'],
            obj_anno[prefix + '_y'],
            obj_anno[prefix + '_z'],
        ])
    return {
        'objects': {'{:02}'.format(obj_label): {
            'keypoints': rows2array(obj_anno, 'kp'),
            'kp_normals': rows2array(obj_anno, 'kp_normals'),
        } for obj_label, obj_anno in models_info.items()},
    }


def get_dataset(configs, mode):
    return SixdDataset(configs, mode)


Annotation = namedtuple('Annotation', ['cls', 'group_id', 'bbox2d', 'keypoint', 'location', 'rotation'])


class SixdDataset(Dataset):
    def __init__(self, configs, mode):
        self._configs = configs.data
        self._yaml_dict = {}
        self._metadata = get_metadata(configs)
        self._mode = mode
        self._class_map = ClassMap(configs)
        self._gt_map_generator = GtMapsGenerator(configs)
        self._sequence_lengths = self._init_sequence_lengths()
        self._models = self._init_models()

    def _read_yaml(self, path):
        if path not in self._yaml_dict:
            with open(path, 'r') as f:
                self._yaml_dict[path] = yaml.load(f, Loader=yaml.CLoader)
        return self._yaml_dict[path]

    def _init_sequence_lengths(self):
        sequences = OrderedDict()
        root_path = self._configs.path
        for sequence in self._configs.sequences[self._mode]:
            num_images = len(listdir_nohidden(join(root_path, sequence, 'rgb')))
            sequences[sequence] = num_images
        return sequences

    def _init_models(self):
        return self._read_yaml(join(self._configs.path, 'models', 'models_info.yml'))

    def __len__(self):
        return sum(self._sequence_lengths.values())

    def __getitem__(self, index):
        #index = int(index)
        seq_name, img_ind = self._get_data_pointers(index)
        dir_path = join(self._configs.path, seq_name)
        data = self._read_data(dir_path, img_ind)
        annotations = self._read_annotations(dir_path, img_ind)
        calibration = self._read_calibration(dir_path, img_ind)
        gt_maps = self._mode in (TRAIN, VAL) and \
                  self._gt_map_generator.generate(annotations, calibration)
        return Sample(annotations, data, gt_maps, calibration, index)

    def _read_data(self, dir_path, img_ind):
        path = join(dir_path, 'rgb', str(img_ind).zfill(4) + '.png')
        image = read_image_to_pt(path)
        max_h, max_w = self._configs.img_dims
        return image[:, :max_h, :max_w]

    def _read_annotations(self, dir_path, img_ind):
        annotations = []
        gts = self._read_yaml(join(dir_path, 'gt.yml'))[img_ind]
        calib = self._read_calibration(dir_path, img_ind)
        for gt in gts:
            model = self._models[gt['obj_id']]

            group_label = self._class_map.format_group_label(gt['obj_id'])
            group_id = self._class_map.group_id_from_group_label(group_label)

            location = Tensor(gt['cam_t_m2c'])
            rot_matrix = np.array(gt['cam_R_m2c']).reshape((3, 3))
            keypoints_3d = self._metadata['objects'][group_label]['keypoints']
            keypoints_2d = project_3d_pts(
                keypoints_3d,
                calib,
                location,
                rot_matrix=rot_matrix,
            )
            kp_normals_global_frame = rot_matrix @ self._metadata['objects'][group_label]['kp_normals']
            for kp_idx in range(NBR_KEYPOINTS):
                normal_pointing_away = kp_normals_global_frame[2,kp_idx] > 0.0
                if normal_pointing_away:
                    continue

                x1 = int(keypoints_2d[0,kp_idx]) - PATCH_SIZE//2
                x2 = int(keypoints_2d[0,kp_idx]) + PATCH_SIZE//2
                y1 = int(keypoints_2d[1,kp_idx]) - PATCH_SIZE//2
                y2 = int(keypoints_2d[1,kp_idx]) + PATCH_SIZE//2
                bbox2d = Tensor([x1, y1, x2, y2])
                class_id = self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx)
                annotations.append(Annotation(cls=class_id,
                                              group_id=group_id,
                                              bbox2d=bbox2d,
                                              keypoint=keypoints_2d[:,kp_idx],
                                              # All patches share the R, t annotation:
                                              location=location,
                                              rotation=rot_matrix,
                                          ))
        return annotations

    def _read_calibration(self, dir_path, img_ind):
        obj_info = self._read_yaml(join(dir_path, 'info.yml'))[img_ind]
        intrinsic = np.reshape(obj_info['cam_K'], (3, 3))
        return np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)

    def _get_data_pointers(self, index):
        for seq_name, seq_length in self._sequence_lengths.items():
            if seq_length > index:
                break
            index -= seq_length
        return seq_name, index

class ClassMap:
    """Mapping between class label and class id.

    id (int): the class index in the network layer.
    label (str): the name of the class

    """
    def __init__(self, configs):
        with open(join(configs.data.path, 'models', 'models_info.yml'), 'r') as model_file:
            group_labels_int = sorted(yaml.load(model_file, Loader=yaml.CLoader).keys())
        group_labels_str = list(map(self.format_group_label, group_labels_int))
        group_ids = list(range(0, len(group_labels_str)))
        self._group_label2group_id_dict = dict(list(zip(group_labels_str, group_ids)))
        self._group_id2group_label_dict = dict(list(zip(group_ids, group_labels_str)))

        class_labels_str = ["obj{:s}_kp{:02d}".format(group_label, kp_idx) for group_label in group_labels_str for kp_idx in range(NBR_KEYPOINTS)]
        # In network, 0 and 1 are reserved for background and don't_care
        class_ids = list(range(2, len(class_labels_str)+2))
        self._class_label2class_id_dict = dict(list(zip(class_labels_str, class_ids)))
        self._class_id2class_label_dict = dict(list(zip(class_ids, class_labels_str)))

        group_id_and_kp_idx_tuples = [(group_id, kp_idx) for group_id in group_ids for kp_idx in range(NBR_KEYPOINTS)]
        self._group_id_and_kp_idx2class_id_dict = dict(list(zip(group_id_and_kp_idx_tuples, class_ids)))
        self._class_id2group_id_and_kp_idx_dict = dict(list(zip(class_ids, group_id_and_kp_idx_tuples)))

    # CLASS ID METHODS ---------------------------------------------------------
    def id_from_label(self, class_label):
        return self._class_label2class_id_dict[class_label]

    def label_from_id(self, class_id):
        return self._class_id2class_label_dict[class_id]

    def get_ids(self):
        return sorted(self._class_id2class_label_dict.keys())

    def get_color(self, class_id):
        if isinstance(class_id, str):
            class_id = self.id_from_label(class_id)
        return cm.Set3(class_id % 12)

    # GROUP ID METHODS ---------------------------------------------------------
    def format_group_label(self, group_label_int):
        group_label_str = "{:02d}".format(group_label_int)
        return group_label_str

    def group_id_from_group_label(self, group_label):
        return self._group_label2group_id_dict[group_label]

    def group_label_from_group_id(self, group_id):
        return self._group_id2group_label_dict[group_id]

    def get_group_ids(self):
        return sorted(self._group_id2group_label_dict.keys())

    # CLASS / GROUP ID METHODS -------------------------------------------------
    def class_id_from_group_id_and_kp_idx(self, group_id, kp_idx):
        return self._group_id_and_kp_idx2class_id_dict[(group_id, kp_idx)]

    def group_id_and_kp_idx_from_class_id(self, class_id):
        group_id, kp_idx = self._class_id2group_id_and_kp_idx_dict[class_id]
        return (group_id, kp_idx)
