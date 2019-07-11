"""Reading input data in the SIXD common format."""
from os.path import join
from collections import namedtuple, OrderedDict
import yaml
import os
import shutil
import glob

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from matplotlib.pyplot import cm

from lib.constants import IGNORE_IDX_CLS, TRAIN, VAL
from lib.data.loader import Sample
from lib.data.maps import GtMapsGenerator
from lib.utils import read_image_to_pt
from lib.utils import listdir_nohidden
from lib.utils import get_metadata
from lib.utils import project_3d_pts
from lib.utils import read_yaml_and_pickle


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


Annotation = namedtuple('Annotation', ['cls', 'bbox2d', 'keypoints', 'kp_visibility', 'size', 'location', 'rotation'])


class SixdDataset(Dataset):
    def __init__(self, configs, mode):
        self._configs = configs
        self._yaml_dict = {}
        self._metadata = get_metadata(configs)
        self._mode = mode
        self._class_map = ClassMap(configs)
        # configs['data']['class_map'] = self._class_map
        self._gt_map_generator = GtMapsGenerator(configs)
        self._sequence_lengths = self._init_sequence_lengths()
        if self._mode == TRAIN:
            self._extra_sequence_lengths = self._init_extra_sequence_lengths()
        self._models = self._init_models()

    def _read_yaml(self, path):
        return read_yaml_and_pickle(path)
        # if path not in self._yaml_dict:
        #     with open(path, 'r') as f:
        #         self._yaml_dict[path] = yaml.load(f, Loader=yaml.CLoader)
        # return self._yaml_dict[path]

    def _check_seq_has_annotations_of_interest(self, root_path, sequence):
        global_info_path = join(root_path, sequence, 'global_info.yml')
        if os.path.exists(global_info_path):
            global_info_yaml = self._read_yaml(global_info_path)
            objs_of_interest = sorted(self._class_map._label2id_dict.keys())
            if len(set(objs_of_interest) & set(global_info_yaml['obj_annotated_and_present'])) == 0:
                print('No annotations for objects of interest {} in sequence, discarding: {}'.format(objs_of_interest, sequence))
                return False
        return True

    def _init_sequence_lengths(self):
        sequences = OrderedDict()
        root_path = self._configs.data.path
        seqs = [seq for seq_pattern in self._configs.data.sequences[self._mode] for seq in seq_glob_expand(self._configs, seq_pattern)] # Parse & expand glob patterns
        for sequence in seqs:
            if not self._check_seq_has_annotations_of_interest(root_path, sequence):
                continue
            num_images = len(listdir_nohidden(join(root_path, sequence, 'rgb')))
            sequences[sequence] = num_images
        assert len(sequences) > 0
        return sequences

    def _init_extra_sequence_lengths(self):
        sequences = OrderedDict()
        root_path = self._configs.data.path
        # seqs = [seq for seq_pattern in self._configs.data.sequences.train_extra for seq in seq_glob_expand(self._configs, seq_pattern)] # Parse & expand glob patterns
        seqs = [spec['seq_name'] for spec in self._configs.data.sequences.train_extra]
        for sequence in seqs:
            if not self._check_seq_has_annotations_of_interest(root_path, sequence):
                continue
            num_images = len(listdir_nohidden(join(root_path, sequence, 'rgb')))
            sequences[sequence] = num_images
        return sequences

    def _init_models(self):
        return self._read_yaml(join(self._configs.data.path, 'models', 'models_info.yml'))

    def __len__(self):
        return sum(self._sequence_lengths.values())

    def __getitem__(self, index):
        #index = int(index)
        seq_name, img_ind = self._get_data_pointers(index)
        dir_path = join(self._configs.data.path, seq_name)
        data = self._read_data(dir_path, img_ind)
        calibration = self._read_calibration(dir_path, img_ind)
        annotations = self._read_annotations(dir_path, img_ind, calibration)
        gt_maps = self._mode in (TRAIN, VAL) and \
                  self._gt_map_generator.generate(annotations, calibration)
        return Sample(annotations, data, gt_maps, calibration, index)

    def _read_data(self, dir_path, img_ind):
        path = join(dir_path, 'rgb', str(img_ind).zfill(6) + '.png')
        image = read_image_to_pt(path)
        max_h, max_w = self._configs.data.img_dims
        return image[:, :max_h, :max_w]

    def _read_annotations(self, dir_path, img_ind, calib):
        annotations = []
        gts = self._read_yaml(join(dir_path, 'gt.yml'))[img_ind]
        for gt in gts:
            model = self._models[gt['obj_id']]
            class_label = self._class_map.sixd_obj_id2class_label(gt['obj_id'])
            if class_label not in self._class_map._label2id_dict.keys():
                continue
            class_id = self._class_map.id_from_label(class_label)

            location = Tensor(gt['cam_t_m2c'])
            rot_matrix = np.array(gt['cam_R_m2c']).reshape((3, 3))

            keypoints_3d = self._metadata['objects'][class_label]['keypoints']
            keypoints_2d = project_3d_pts(
                keypoints_3d,
                calib,
                location,
                rot_matrix=rot_matrix,
            )

            kp_normals_global_frame = rot_matrix @ self._metadata['objects'][class_label]['kp_normals']
            kp_visibility = kp_normals_global_frame[2,:] <= 0.0 # neg z-coordinate => normal points towards camera

            bbox2d = Tensor(gt['obj_bb'])
            bbox2d[2:] += bbox2d[:2]  # x,y,w,h, -> x1,y1,x2,y2
            # Size uses KITTI convention - when rot=0 we have that: h, w, l  <-->  y, z, x
            size = Tensor((model['size_y'], model['size_z'], model['size_x']))
            annotations.append(Annotation(cls=class_id,
                                          bbox2d=bbox2d,
                                          keypoints=keypoints_2d,
                                          kp_visibility=kp_visibility,
                                          size=size,
                                          location=location,
                                          rotation=rot_matrix))
        return annotations

    def _read_calibration(self, dir_path, img_ind):
        obj_info = self._read_yaml(join(dir_path, 'info.yml'))[img_ind]
        intrinsic = np.reshape(obj_info['cam_K'], (3, 3))
        return np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)

    def _get_data_pointers(self, index):
        if self._mode == TRAIN and len(self._configs.data.sequences.train_extra) > 0:
            seq_names, sample_probs = zip(*[(spec['seq_name'], spec['sample_prob']) for spec in self._configs.data.sequences.train_extra])
            seq_idx = np.random.choice(len(sample_probs)+1, p = [1.0-sum(sample_probs)]+list(sample_probs))
            if seq_idx > 0:
                # seq_idx == 0 corresponds to not sampling from the "extra" sequences
                seq_name = seq_names[seq_idx-1]
                index = np.random.choice(self._extra_sequence_lengths[seq_name])
                # print(seq_name)
                return seq_name, index
        for seq_name, seq_length in self._sequence_lengths.items():
            if seq_length > index:
                break
            index -= seq_length
        return seq_name, index


# class ClassMap:
#     """ClassMap."""
#     def __init__(self, configs):
#         self._cls_dict = configs.data.class_map
# 
#     def id_from_label(self, label):
#         return self._cls_dict.get(label, IGNORE_IDX_CLS)
# 
#     def label_from_id(self, class_id):
#         return next(label for label, id_ in self._cls_dict.items() if id_ is class_id)
# 
#     def get_ids(self):
#         return set(self._cls_dict.values()) - {IGNORE_IDX_CLS}
# 
#     def get_color(self, class_id):
#         if isinstance(class_id, str):
#             class_id = self.id_from_label(class_id)
#         return cm.Set3(class_id % 12)

class ClassMap:
    """Mapping between class label and class id.

    id (int): the class index in the network layer.
    label (str): the name of the class

    """
    def __init__(self, configs):
        with open(join(configs.data.path, 'models', 'models_info.yml'), 'r') as model_file:
            self._models_info = yaml.load(model_file, Loader=yaml.CLoader)
            class_labels_int = sorted(self._models_info.keys())
        class_labels_str = list(map(self.sixd_obj_id2class_label, class_labels_int))
        if configs.data.class_labels is not None:
            assert set(configs.data.class_labels) <= set(class_labels_str)
            class_labels_str = configs.data.class_labels

        # In network, 0 and 1 are reserved for background and don't_care
        class_ids = list(range(1, len(class_labels_str)+1))
        # class_ids = list(range(2, len(class_labels_str)+2))
        self._label2id_dict = dict(list(zip(class_labels_str, class_ids)))
        self._id2label_dict = dict(list(zip(class_ids, class_labels_str)))

    def sixd_obj_id2class_label(self, sixd_obj_id):
        class_label_str = self._models_info[sixd_obj_id]['readable_label']
        return class_label_str

    def id_from_label(self, class_label):
        return self._label2id_dict[class_label]

    def label_from_id(self, class_id):
        return self._id2label_dict[class_id]

    def get_ids(self):
        return sorted(self._id2label_dict.keys())

    def get_color(self, class_id):
        if isinstance(class_id, str):
            class_id = self.id_from_label(class_id)
        return cm.Set3(class_id % 12)
