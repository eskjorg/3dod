"""Generate GT maps for training."""
import sys
from abc import ABCMeta, abstractmethod

from math import ceil
import numpy as np
import torch

from lib.constants import IGNORE_IDX_CLS, KEYPOINT_NAME_MAP
from lib.utils import get_layers, project_3d_box, matrix_from_yaw


class GtMapsGenerator:
    """GtMapsGenerator."""
    def __init__(self, configs):
        super(GtMapsGenerator, self).__init__()
        self._configs = configs
        self._configs.target_dims = [ceil(dim / self._configs.network.output_stride)
                                     for dim in self._configs.data.img_dims]
        self._layers = get_layers(self._configs.config_name)

    def generate(self, annotations, calibration):
        gt_maps = {}
        obj_coords_full = self._get_coordinates(annotations)
        obj_coords_supp = self._get_coordinates(annotations, self._configs.network.support_region)
        for layer_name in self._layers.keys():
            Generator = getattr(sys.modules[__name__], layer_name.capitalize() + 'Generator')
            generator = Generator(self._configs, calibration)
            for obj, supp, full in zip(annotations, obj_coords_supp, obj_coords_full):
                if layer_name == "class":
                    generator.add_obj(obj, full, IGNORE_IDX_CLS)
                if obj.cls is not IGNORE_IDX_CLS:
                    generator.add_obj(obj, supp)
            gt_maps[layer_name] = generator.get_map()
        return gt_maps

    def _get_coordinates(self, objects, shrink_factor=1):
        obj_coords = []
        target_dims = self._configs.target_dims
        def get_coord_convex(coord1, coord2, max_limit):
            coord = coord1 * (1 + shrink_factor) / 2 + \
                    coord2 * (1 - shrink_factor) / 2
            return  int(max(0, min(max_limit, coord)))
        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox2d / self._configs.network.output_stride
            obj_coords.append((get_coord_convex(xmin, xmax, target_dims[1]),
                               get_coord_convex(ymin, ymax, target_dims[0]),
                               get_coord_convex(xmax, xmin, target_dims[1]),
                               get_coord_convex(ymax, ymin, target_dims[0])))
        return obj_coords


class GeneratorIf(metaclass=ABCMeta):
    """Abstract class for target gt generation."""
    def __init__(self, configs, *args, **kwargs):
        """Constructor."""
        super().__init__()
        self._configs = configs
        self._map = torch.zeros(self._get_num_maps(), *self._configs.target_dims)

    @abstractmethod
    def _get_num_maps(self):
        """Specify how many outputs this map generator supports."""

    @abstractmethod
    def add_obj(self, obj_annotation, map_coords):
        """Add a single object to the maps."""

    def get_map(self):
        """Generate a network target from the `objects`."""
        return self._map

    def decode(self, tensor):
        """Decode an output tensor."""
        return tensor


class GeneratorIndex(GeneratorIf):
    """GeneratorIndex."""
    def __init__(self, configs, calib=None, device='cpu'):
        """Constructor."""
        super().__init__(configs)
        self._calib = calib
        self._index_map = self._gen_index_map().to(torch.device(device))

    def _gen_index_map(self):
        img_height, img_width = self._configs.data.img_dims
        stride = self._configs.network.output_stride
        map_height = ceil(img_height / stride)
        map_width = ceil(img_width / stride)
        return torch.from_numpy(stride * np.indices((map_height, map_width), dtype=np.float32))

    def decode(self, tensor):
        tensor[0::2, :] += self._index_map[1]
        tensor[1::2, :] += self._index_map[0]
        return tensor


class ClsGenerator(GeneratorIf):
    """GT map class generator."""
    def _get_num_maps(self):
        return 1

    def add_obj(self, obj_annotation, map_coords, obj_class=None):
        xmin, ymin, xmax, ymax = map_coords
        self._map[0, ymin: ymax, xmin: xmax] = obj_class or obj_annotation.cls

    def get_map(self):
        return self._map.long()


class Bbox2dGenerator(GeneratorIndex):
    """GT map Bbox2dGenerator."""
    def _get_num_maps(self):
        return 4

    def add_obj(self, obj_annotation, map_coords):
        xmin, ymin, xmax, ymax = map_coords
        bbox_coords = obj_annotation.bbox2d
        for index in range(self._get_num_maps()):
            self._map[index, ymin: ymax, xmin: xmax] = \
                bbox_coords[index] - self._index_map[(index + 1) % 2, ymin: ymax, xmin: xmax]


class ZdepthGenerator(GeneratorIf):
    """GT map ZdepthGenerator."""
    def _get_num_maps(self):
        return 1

    def add_obj(self, obj_annotation, map_coords):
        xmin, ymin, xmax, ymax = map_coords
        zdepth = obj_annotation.location[2]
        self._map[0, ymin: ymax, xmin: xmax] = zdepth


class SizeGenerator(GeneratorIf):
    """GT map SizeGenerator."""
    def _get_num_maps(self):
        return 3

    def add_obj(self, obj_annotation, map_coords):
        log_dim = torch.Tensor(obj_annotation.size).log()
        xmin, ymin, xmax, ymax = map_coords
        for index in range(self._get_num_maps()):
            self._map[index, ymin: ymax, xmin: xmax] = log_dim[index]

    def decode(self, tensor):
        return tensor.exp()


class CornersGenerator(GeneratorIndex):
    """GT map CornersGenerator."""
    def _get_num_maps(self):
        return 16

    def add_obj(self, obj_annotation, map_coords):
        xmin, ymin, xmax, ymax = map_coords
        rotation = matrix_from_yaw(obj_annotation.rot_y) if hasattr(obj_annotation, 'rot_y') \
                   else obj_annotation.rotation
        corner_coords = project_3d_box(self._calib,
                                       obj_annotation.size,
                                       obj_annotation.location,
                                       rot_matrix=rotation)
        for index in range(8):
            x, y = corner_coords[:, index]
            # Quite arbitrary threshold: 10
            # If keypoint is slightly outside of image is OK
            # If keypoint is far off, it will hurt training
            if abs(x / self._configs.data.img_dims[1] - 0.5) > 10:
                print('Keypoint projected far outside image. Check object not behind camera.')
            self._map[2 * index + 0, ymin: ymax, xmin: xmax] = x - self._index_map[1, ymin: ymax, xmin: xmax]
            self._map[2 * index + 1, ymin: ymax, xmin: xmax] = y - self._index_map[0, ymin: ymax, xmin: xmax]
