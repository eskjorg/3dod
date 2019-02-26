"""Reading input data in the SIXD common format."""
from os.path import join
from collections import namedtuple, OrderedDict
import yaml

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from matplotlib.pyplot import cm

from lib.constants import IGNORE_IDX_CLS, TRAIN, VAL
from lib.data.loader import Sample
from lib.data.maps import GtMapsGenerator
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


def get_dataset(configs, mode):
    return SixdDataset(configs, mode)


Annotation = namedtuple('Annotation', ['cls', 'bbox2d', 'size', 'location', 'rotation'])


class SixdDataset(Dataset):
    def __init__(self, configs, mode):
        self._configs = configs.data
        self._mode = mode
        self._class_map = ClassMap(configs)
        self._gt_map_generator = GtMapsGenerator(configs)
        self._sequence_lengths = self._init_sequence_lengths()
        self._models = self._init_models()

    def _init_sequence_lengths(self):
        sequences = OrderedDict()
        root_path = self._configs.path
        for sequence in self._configs.sequences[self._mode]:
            num_images = len(listdir_nohidden(join(root_path, sequence, 'rgb')))
            sequences[sequence] = num_images
        return sequences

    def _init_models(self):
        path = join(self._configs.path, 'models', 'models_info.yml')
        with open(path, 'r') as file:
            return yaml.load(file)

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
        with open(join(dir_path, 'gt.yml'), 'r') as file:
            gts = yaml.load(file, Loader=yaml.CLoader)[img_ind]
        for gt in gts:
            model = self._models[gt['obj_id']]
            bbox2d = Tensor(gt['obj_bb'])
            bbox2d[2:] += bbox2d[:2]  # x,y,w,h, -> x1,y1,x2,y2
            # Size uses KITTI convention - when rot=0 we have that: h, w, l  <-->  y, z, x
            size = Tensor((model['size_y'], model['size_z'], model['size_x']))
            annotations.append(Annotation(cls=self._class_map.id_from_label(self._class_map.format_label(gt['obj_id'])),
                                          bbox2d=bbox2d,
                                          size=size,
                                          location=Tensor(gt['cam_t_m2c']),
                                          rotation=np.array(gt['cam_R_m2c']).reshape((3, 3))))
        return annotations

    def _read_calibration(self, dir_path, img_ind):
        with open(join(dir_path, 'info.yml'), 'r') as file:
            obj_info = yaml.load(file, Loader=yaml.CLoader)[img_ind]
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
            class_labels_int = sorted(yaml.load(model_file).keys())
        class_labels_str = list(map(self.format_label, class_labels_int))
        # In network, 0 and 1 are reserved for background and don't_care
        class_ids = list(range(2, len(class_labels_str)+2))
        self._label2id_dict = dict(list(zip(class_labels_str, class_ids)))
        self._id2label_dict = dict(list(zip(class_ids, class_labels_str)))

    def format_label(self, class_label_int):
        class_label_str = "{:02d}".format(class_label_int)
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
