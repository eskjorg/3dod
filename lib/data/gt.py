"""Generate GT masks for training."""

import sys
from abc import ABCMeta, abstractmethod

from math import ceil
import numpy as np

from lib.constants import IGNORE_IDX_CLS, KEYPOINT_NAME_MAP
from lib.utils import get_layers, project_3d_box


class GTGenerator:
    """GTGenerator."""
    def __init__(self, configs):
        super(GTGenerator, self).__init__()
        self._configs = configs
        self._configs.target_dims = [ceil(dim / self._configs.network.output_stride)
                                     for dim in self._configs.data.img_dims]
        self._layers = get_layers(self._configs.config_load_path)

    def generate(self, annotations, calibration):
        gt_masks = {}
        obj_coords_full = self._get_coordinates(annotations)
        obj_coords_supp = self._get_coordinates(annotations, self._configs.network.support_region)
        for layer_name in self._layers.keys():
            Generator = getattr(sys.modules[__name__], layer_name.capitalize() + 'Generator')
            generator = Generator(self._configs, calibration.P0)
            for object, supp, full in zip(annotations, obj_coords_supp, obj_coords_full):
                if layer_name == "class":
                    generator.add_obj(object, full)
                if object.obj_class is not IGNORE_IDX_CLS:
                    generator.add_obj(object, supp)
            gt_masks[layer_name] = generator.get_mask()

    def _get_coordinates(self, objects, shrink_factor=1):
        obj_coords = []
        target_dims = self._configs.target_dims
        def get_coord_convex(coord1, coord2, max_limit):
            coord = coord1 * (1 + shrink_factor) / 2 + \
                    coord2 * (1 - shrink_factor) / 2
            return  int(max(0, min(max_limit, coord)))
        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bounding_box / self._configs.network.output_stride
            obj_coords.append((get_coord_convex(xmin, xmax, target_dims[1]),
                               get_coord_convex(ymin, ymax, target_dims[0]),
                               get_coord_convex(xmax, xmin, target_dims[1]),
                               get_coord_convex(ymax, ymin, target_dims[0])))
        return obj_coords


class GeneratorIf(metaclass=ABCMeta):
    """Abstract class for target gt generation."""
    def __init__(self, configs, *args):
        """Constructor."""
        super().__init__()
        self._configs = configs
        self._mask = np.zeros((self._get_num_masks(), *self._configs.target_dims))

    @abstractmethod
    def _get_num_masks(self):
        """Specify how many outputs this mask generator supports."""

    @abstractmethod
    def add_obj(self, obj_annotation, mask_coords):
        """Add a single object to the masks."""

    def get_mask(self):
        """Generate a network target from the `objects`."""
        return self._mask


class ClassGenerator(GeneratorIf):
    """GT mask ClassGenerator."""
    def _get_num_masks(self):
        return 1

    def add_obj(self, obj_annotation, mask_coords, obj_class=None):
        xmin, ymin, xmax, ymax = mask_coords
        self._mask[ymin: ymax, xmin: xmax] = obj_class or obj_annotation.obj_class

    def get_mask(self):
        return self._mask.astype('uint8')


class Bbox2dGenerator(GeneratorIf):
    """GT mask Bbox2dGenerator."""
    def _get_num_masks(self):
        return 4

    def add_obj(self, obj_annotation, mask_coords):
        xmin, ymin, xmax, ymax = mask_coords
        bbox_coords = obj_annotation.bounding_box
        for index in range(self._get_num_masks()):
            self._mask[index, ymin: ymax, xmin: xmax] = bbox_coords[index]

    def get_mask(self):
        bbox_mask_handler = IndexCodec(*self._configs.data.img_dims, self._configs.network.output_stride)
        return bbox_mask_handler.encode(self._mask)


class ZdepthGenerator(GeneratorIf):
    """GT mask ZdepthGenerator."""
    def _get_num_masks(self):
        return 1

    def add_obj(self, obj_annotation, mask_coords):
        xmin, ymin, xmax, ymax = mask_coords
        zdepth = obj_annotation.location[2]
        self._mask[0, ymin: ymax, xmin: xmax] = zdepth


class SizeGenerator(GeneratorIf):
    """GT mask SizeGenerator."""
    def _get_num_masks(self):
        return 3

    def add_obj(self, obj_annotation, mask_coords):
        log_dim = np.log(np.abs(obj_annotation.dimensions))
        xmin, ymin, xmax, ymax = mask_coords
        for index in range(self._get_num_masks()):
            self._mask[index, ymin: ymax, xmin: xmax] = log_dim[index]


class CornersGenerator(GeneratorIf):
    """GT mask CornersGenerator."""
    def __init__(self, configs, projection_matrix):
        """Constructor."""
        super().__init__(configs)
        self._projection_matrix = projection_matrix

    def _get_num_masks(self):
        return 16

    def add_obj(self, obj_annotation, mask_coords):
        xmin, ymin, xmax, ymax = mask_coords
        corner_coords = project_3d_box(obj_annotation.dimensions,
                                       obj_annotation.location,
                                       obj_annotation.rotation,
                                       self._projection_matrix)
        for index in range(8):
            x, y = corner_coords[:, index]
            # Quite arbitrary threshold: 10
            # If keypoint is slightly outside of image is OK
            # If keypoint is far off, it will hurt training
            if abs(x / self._configs.data.img_dims[1] - 0.5) > 10:
                print('Keypoint projected far outside image. Check object not behind camera.')
            self._mask[2 * index + 0, ymin: ymax, xmin: xmax] = x
            self._mask[2 * index + 1, ymin: ymax, xmin: xmax] = y

    def get_mask(self):
        corner_mask_handler = IndexCodec(*self._configs.data.img_dims, self._configs.network.output_stride)
        return corner_mask_handler.encode(self._mask)


class IndexCodec():
    def __init__(self, img_height, img_width, stride):
        """Constructor."""
        mask_height = ceil(img_height / stride)
        mask_width = ceil(img_width / stride)

        self._index_matrix_x = np.zeros([mask_height, mask_width])
        self._index_matrix_y = np.zeros([mask_height, mask_width])

        for i in range(mask_height):
            self._index_matrix_x[i, :] = np.arange(0, img_width, stride)
        for i in range(mask_width):
            self._index_matrix_y[:, i] = np.arange(0, img_height, stride)

    def encode(self, mask):
        mask[0::2, :] -= self._index_matrix_x
        mask[1::2, :] -= self._index_matrix_y
        return mask

    def decode(self, mask):
        mask[0::2, :] += self._index_matrix_x
        mask[1::2, :] += self._index_matrix_y
        return mask
