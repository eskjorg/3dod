"""Reading input data in the SIXD common format."""
from os.path import join
from collections import namedtuple, OrderedDict
import yaml
import os
import glob

import math
import cv2 as cv
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from matplotlib.pyplot import cm

from lib.constants import IGNORE_IDX_CLS, TRAIN, VAL, NBR_KEYPOINTS, PATCH_SIZE
from lib.data.loader import Sample
from lib.data.maps import GtMapsGenerator
from lib.utils import read_image_to_pt
from lib.utils import read_seg_to_pt
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
    def get_bbox3d(min_bounds, diff):
        max_bounds = min_bounds + diff
        return np.hstack([min_bounds[:,None], max_bounds[:,None]])
    return {
        'objects': {obj_anno['readable_label']: {
            'bbox3d': get_bbox3d(rows2array(obj_anno, 'min'), rows2array(obj_anno, 'size')),
            'keypoints': rows2array(obj_anno, 'kp'),
            'kp_normals': rows2array(obj_anno, 'kp_normals'),
        } for obj_label, obj_anno in models_info.items()},
    }


def get_dataset(configs, mode):
    return SixdDataset(configs, mode)


def seq_glob_expand(configs, seq_pattern):
    """
    Find multiple sequences from wildcard pattern
    """
    seqs = glob.glob(os.path.join(configs.data.path, seq_pattern))
    seqs = [os.path.relpath(absolute_path, configs.data.path) for absolute_path in seqs]
    return seqs


Annotation = namedtuple('Annotation', ['cls', 'group_id', 'bbox2d', 'keypoint', 'keypoint_detectability', 'self_occluded', 'occluded', 'location', 'rotation'])


class SixdDataset(Dataset):
    def __init__(self, configs, mode):
        self._configs = configs
        assert self._configs.data.img_dims[0] % self._configs.network.output_stride == 0
        assert self._configs.data.img_dims[1] % self._configs.network.output_stride == 0
        self._yaml_dict = {}
        self._metadata = get_metadata(configs)
        self._mode = mode
        self._class_map = ClassMap(configs)
        self._gt_map_generator = GtMapsGenerator(configs)
        self._sequence_lengths = self._init_sequence_lengths()
        self._models = self._init_models()
        self._gdists = self._init_gdists()

    def _read_yaml(self, path):
        if path not in self._yaml_dict:
            with open(path, 'r') as f:
                self._yaml_dict[path] = yaml.load(f, Loader=yaml.CLoader)
        return self._yaml_dict[path]

    def _init_sequence_lengths(self):
        sequences = OrderedDict()
        root_path = self._configs.data.path
        seqs = [seq for seq_pattern in self._configs.data.sequences[self._mode] for seq in seq_glob_expand(self._configs, seq_pattern)] # Parse & expand glob patterns
        for sequence in seqs:
            num_images = len(listdir_nohidden(join(root_path, sequence, 'rgb')))
            sequences[sequence] = num_images
        return sequences

    def _init_models(self):
        return self._read_yaml(join(self._configs.data.path, 'models', 'models_info.yml'))

    def _init_gdists(self):
        return self._read_yaml(join(self._configs.data.path, 'models', 'gdists.yml'))

    def __len__(self):
        return sum(self._sequence_lengths.values())

    # LINEMOD-specific:
    def _lookup_unannotated_class_ids(self, seq_name):
        subset, seq_name = seq_name.split('/')
        if subset in ['train_unoccl', 'train_aug']:
            group_id_annotated = self._class_map.group_id_from_group_label(seq_name)
            return [self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx) for group_id in self._class_map.get_group_ids() for kp_idx in range(NBR_KEYPOINTS) if group_id != group_id_annotated]
        return []

    def __getitem__(self, index):
        #index = int(index)
        seq_name, img_ind = self._get_data_pointers(index)
        dir_path = join(self._configs.data.path, seq_name)
        data = self._read_data(dir_path, img_ind)
        vtx_idx_map = self._read_vtx_idx_map(dir_path, img_ind)
        instance_seg = self._read_instance_seg(dir_path, img_ind)
        seg = self._seg_from_instance_seg(dir_path, img_ind, instance_seg)
        calibration = self._read_calibration(dir_path, img_ind)
        annotations = self._read_annotations(dir_path, img_ind, calibration, instance_seg, vtx_idx_map)
        unannotated_class_ids = self._lookup_unannotated_class_ids(seq_name)
        gt_maps = self._mode in (TRAIN, VAL) and \
                  self._gt_map_generator.generate(annotations, calibration, unannotated_class_ids=unannotated_class_ids, seg=seg)
        return Sample(annotations, data, gt_maps, calibration, index)

    def _read_data(self, dir_path, img_ind):
        path = join(dir_path, 'rgb', str(img_ind).zfill(6) + '.png')
        image = read_image_to_pt(path, load_type=cv.IMREAD_COLOR, normalize_flag=True)
        max_h, max_w = self._configs.data.img_dims
        return image[:, :max_h, :max_w]

    def _read_instance_seg(self, dir_path, img_ind):
        path = join(dir_path, 'instance_seg', str(img_ind).zfill(6) + '.png')
        instance_seg = read_seg_to_pt(path)
        max_h, max_w = self._configs.data.img_dims
        return instance_seg[:max_h, :max_w]

    def _read_vtx_idx_map(self, dir_path, img_ind):
        path = join(dir_path, 'vtx_idx', str(img_ind).zfill(6) + '.png')
        vtx_idx_map = Image.open(path)
        vtx_idx_map = np.array(vtx_idx_map)
        max_h, max_w = self._configs.data.img_dims
        return vtx_idx_map[:max_h, :max_w]

    def _seg_from_instance_seg(self, dir_path, img_ind, instance_seg):
        seg = torch.zeros_like(instance_seg, dtype=torch.uint8)
        gts = self._read_yaml(join(dir_path, 'gt.yml'))[img_ind]
        instance_idx = 0
        for gt in gts:
            instance_idx += 1
            group_label = self._class_map.sixd_obj_id2group_label(gt['obj_id'])
            if group_label not in self._class_map._group_label2group_id_dict.keys():
                continue
            group_id = self._class_map.group_id_from_group_label(group_label)
            seg[instance_seg == instance_idx] = group_id
        return seg

    def _read_annotations(self, dir_path, img_ind, calib, instance_seg, vtx_idx_map):
        annotations = []
        gts = self._read_yaml(join(dir_path, 'gt.yml'))[img_ind]
        instance_idx = 0
        for gt in gts:
            instance_idx += 1
            model = self._models[gt['obj_id']]

            group_label = self._class_map.sixd_obj_id2group_label(gt['obj_id'])
            if group_label not in self._class_map._group_label2group_id_dict.keys():
                continue
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
                # Discard keypoint if not projected inside image
                x, y = keypoints_2d[:,kp_idx]
                x = int(x)
                y = int(y)
                if x < 0 or x >= self._configs.data.img_dims[1]:
                    continue
                elif y < 0 or y >= self._configs.data.img_dims[0]:
                    continue

                # Determine occlusion / self-occlusion
                occluded = instance_seg[y, x] != instance_idx
                self_occluded = kp_normals_global_frame[2,kp_idx] > 0.0

                x1 = int(keypoints_2d[0,kp_idx] - 0.5*(PATCH_SIZE-1))
                x2 = x1 + (PATCH_SIZE-1) + 1 # Box is up until but not including x2
                y1 = int(keypoints_2d[1,kp_idx] - 0.5*(PATCH_SIZE-1))
                y2 = y1 + (PATCH_SIZE-1) + 1 # Box is up until but not including y2

                stride = self._configs.network.output_stride
                x1 = max(0, math.floor(x1 / stride) * stride)
                x2 = min(self._configs.data.img_dims[1], math.ceil(x2 / stride) * stride)
                y1 = max(0, math.floor(y1 / stride) * stride)
                y2 = min(self._configs.data.img_dims[0], math.ceil(y2 / stride) * stride)

                for val in [x1, x2, y1, y2]:
                    assert val % stride == 0

                assert not x1 == x2 or y1 == y2

                bbox2d = Tensor([x1, y1, x2, y2])

                # Set ground truth visibility in the vicinity of the keypoint position
                # keypoint_detectability = (instance_seg[y1:y2, x1:x2] == instance_idx).type(torch.float32)

                mask = instance_seg[y1:y2, x1:x2].numpy() == instance_idx
                keypoint_detectability = np.zeros((y2-y1, x2-x1))
                vtx_idx_vec = vtx_idx_map[y1:y2, x1:x2][mask]
                gdist_patch = self._gdists[gt['obj_id']][kp_idx][vtx_idx_vec]
                # gdist_sigma = 15.0
                gdist_sigma = 30.0
                keypoint_detectability[mask] = np.exp(-0.5*(gdist_patch/gdist_sigma)**2)
                keypoint_detectability = torch.from_numpy(keypoint_detectability).float() # double -> float

                class_id = self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx)
                annotations.append(Annotation(cls=class_id,
                                              group_id=group_id,
                                              bbox2d=bbox2d,
                                              keypoint=keypoints_2d[:,kp_idx],
                                              keypoint_detectability=keypoint_detectability, # Quantify occlusion
                                              self_occluded=self_occluded, # Annotate self-occlusion
                                              occluded=occluded, # Annotate occlusion by other objects (or outside field of view)
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
            self._models_info = yaml.load(model_file, Loader=yaml.CLoader)
            group_labels_int = sorted(self._models_info.keys())
        group_labels_str = list(map(self.sixd_obj_id2group_label, group_labels_int))

        # LINEMOD-specific:
        # seq_names = [seq_name.split('/')[1] for seq_name in configs.data.sequences.train]
        # group_labels_str = [group_label for group_label in group_labels_str if group_label2seq_name[group_label] in seq_names]
        # # group_labels_str = ['05', '08'] # can & driller
        # # group_labels_str = ['08'] # driller
        # # group_labels_str = ['05'] # can
        # # group_labels_str = ['09'] # duck
        # # group_labels_str = ['01', '05', '06', '09'] # ape, can, cat, duck
        if configs.data.group_labels is not None:
            assert set(configs.data.group_labels) <= set(group_labels_str)
            group_labels_str = configs.data.group_labels
            # TODO: Define metadata of objects annotated in each seq, and filter seqs based on groups
            # TODO: Define metadata of objects annotated in each seq, and filter seqs based on groups
            # TODO: Define metadata of objects annotated in each seq, and filter seqs based on groups
            # TODO: Define metadata of objects annotated in each seq, and filter seqs based on groups
            # TODO: Define metadata of objects annotated in each seq, and filter seqs based on groups

        # In network, 0 and 1 are reserved for background and don't_care
        group_ids = list(range(2, len(group_labels_str)+2))
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
    def sixd_obj_id2group_label(self, sixd_obj_id):
        group_label_str = self._models_info[sixd_obj_id]['readable_label']
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
