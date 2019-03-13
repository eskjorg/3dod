"""Generate GT maps for training."""
import sys
from abc import ABCMeta, abstractmethod

from math import ceil
import numpy as np
import torch

from lib.constants import IGNORE_IDX_CLS, IGNORE_IDX_REG, IGNORE_IDX_CLSNONMUTEX, NBR_KEYPOINTS
from lib.utils import get_layers, get_metadata, get_class_map, project_3d_pts, construct_3d_box, matrix_from_yaw


class GtMapsGenerator:
    """GtMapsGenerator."""
    def __init__(self, configs):
        super(GtMapsGenerator, self).__init__()
        self._configs = configs
        self._metadata = get_metadata(self._configs)
        self._class_map = get_class_map(self._configs)
        self._configs.target_dims = [ceil(dim / self._configs.network.output_stride)
                                     for dim in self._configs.data.img_dims]
        self._layers = get_layers(self._configs.config_name)

    def generate(self, annotations, calibration, unannotated_class_ids=[], seg=None):
        def generate_map_for_head(layer_name, obj_coords_full, obj_coords_supp, cls_id_filter=None, fill_values=None, seg=None):
            if cls_id_filter is not None:
                assert layer_name != "cls"
            Generator = getattr(sys.modules[__name__], layer_name.capitalize() + 'Generator')
            generator = Generator(self._configs, self._metadata, self._class_map, calib=calibration, fill_values=fill_values, device='cpu')
            if layer_name == 'clsnonmutex' and not self._layers['clsnonmutex']['ignore_bg'] and len(unannotated_class_ids) > 0:
                # fullres binary mask
                fg_mask_fullres = (seg > 0).unsqueeze(0)
                downsampled = torch.nn.functional.avg_pool2d(
                    fg_mask_fullres.type(torch.float32),
                    self._configs.network.output_stride,
                    stride=self._configs.network.output_stride,
                )
                # lowres mask, requiring 70% foreground pixels
                fg_mask = downsampled > 0.7
                # Initialize all channels with background wherever there is actually an object annotated:
                generator._map[fg_mask.repeat(len(self._class_map.get_ids()), 1, 1)] = 0
            elif layer_name == 'clsnonmutex' and self._layers['clsnonmutex']['ignore_bg']:
                for group_id in self._class_map.get_group_ids():
                    # fullres binary mask
                    fg_mask_fullres = (seg == group_id+1).unsqueeze(0)
                    downsampled = torch.nn.functional.avg_pool2d(
                        fg_mask_fullres.type(torch.float32),
                        self._configs.network.output_stride,
                        stride=self._configs.network.output_stride,
                    )
                    # lowres mask, requiring 70% foreground pixels
                    fg_mask = downsampled > 0.7
                    # Initialize all keypoints with background wherever their corresponding object is annotated:
                    class_ids = np.array([self._class_map.class_id_from_group_id_and_kp_idx(group_id, kp_idx) for kp_idx in range(NBR_KEYPOINTS)])
                    generator._map[class_ids-2][fg_mask.repeat(NBR_KEYPOINTS, 1, 1)] = 0
            for obj, supp, full in zip(annotations, obj_coords_supp, obj_coords_full):
                if cls_id_filter is not None and obj.cls not in cls_id_filter:
                    # Discard instance if head is dedicated only to other class labels
                    continue

                if layer_name == "cls":
                    generator.add_obj(obj, full, IGNORE_IDX_CLS)
                if obj.cls is not IGNORE_IDX_CLS:
                    generator.add_obj(obj, supp)
            return generator.get_map()

        gt_maps = {}
        obj_coords_full = self._get_coordinates(annotations)
        obj_coords_supp = self._get_coordinates(annotations, self._configs.network.support_region)
        for layer_name in self._layers.keys():
            if self._layers[layer_name]['cls_specific_heads']:
                # Separate GT map for every class
                for cls_id in self._class_map.get_ids():
                    class_label = self._class_map.label_from_id(cls_id)
                    gt_maps['{}_{}'.format(layer_name, class_label)] = generate_map_for_head(layer_name, obj_coords_full, obj_coords_supp, cls_id_filter=[cls_id], seg=seg)
            else:
                # Single GT map - shared among all classes
                fill_values = None
                if layer_name == 'clsnonmutex' and self._layers['clsnonmutex']['ignore_bg']:
                    fill_values = IGNORE_IDX_CLSNONMUTEX
                elif layer_name == 'clsnonmutex' and len(unannotated_class_ids) > 0:
                    # print("fill: {}".format(IGNORE_IDX_CLSNONMUTEX))
                    # Avoid using background as negative example - it may hold unannotated objects
                    fill_values = [None] * len(self._class_map.get_ids())
                    for class_id in unannotated_class_ids:
                        fill_values[class_id - 2] = IGNORE_IDX_CLSNONMUTEX
                gt_maps[layer_name] = generate_map_for_head(layer_name, obj_coords_full, obj_coords_supp, cls_id_filter=None, fill_values=fill_values, seg=seg)
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
    def __init__(self, configs, metadata, class_map, *args, fill_values=None, **kwargs):
        """Constructor."""
        super().__init__()
        self._configs = configs
        self._metadata = metadata
        self._class_map = class_map
        self._map = torch.empty(self._get_num_maps(), *configs.target_dims)
        if fill_values is None:
            self._map.fill_(self.fill_value)
        elif hasattr(fill_values, '__iter__'):
            fill_values = [self.fill_value if val is None else val for val in fill_values]
            self._map[:,:,:] = torch.tensor(fill_values).reshape((-1,1,1))
        else:
            self._map.fill_(fill_values)

    def _downsample_keypoint_detectability(self, keypoint_detectability):
        # Easily decimated to output resolution
        assert keypoint_detectability.shape[0] % self._configs.network.output_stride == 0
        assert keypoint_detectability.shape[1] % self._configs.network.output_stride == 0
        keypoint_detectability_ds = torch.nn.functional.avg_pool2d(
            keypoint_detectability.unsqueeze(0),
            self._configs.network.output_stride,
            stride=self._configs.network.output_stride,
        )[0,:,:]
        return keypoint_detectability_ds

    @property
    def fill_value(self):
        # Fill map with IGNORE_IDX_REG by default. Can be overridden
        return IGNORE_IDX_REG

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
    def __init__(self, configs, metadata, class_map, fill_values=None, calib=None, device='cpu'):
        """Constructor."""
        super().__init__(configs, metadata, class_map)
        self._calib = calib
        self._index_map = self._gen_index_map().to(torch.device(device))

    def _gen_index_map(self):
        img_height, img_width = self._configs.data.img_dims
        stride = self._configs.network.output_stride
        assert img_height % stride == 0
        assert img_width % stride == 0
        map_height = img_height // stride
        map_width = img_width // stride
        return torch.from_numpy(stride * np.indices((map_height, map_width), dtype=np.float32))
        # alignment_compensation = self._configs.network.output_stride / 2 - 0.5
        # return torch.from_numpy(stride * np.indices((map_height, map_width), dtype=np.float32)) + alignment_compensation

    def decode(self, tensor):
        tensor[0::2, :] += self._index_map[1]
        tensor[1::2, :] += self._index_map[0]
        return tensor


class ClsGenerator(GeneratorIf):
    """GT map class generator."""
    @property
    def fill_value(self):
        return 0  # Background class

    def _get_num_maps(self):
        return 1

    def add_obj(self, obj_annotation, map_coords, obj_class=None):
        xmin, ymin, xmax, ymax = map_coords
        self._map[0, ymin: ymax, xmin: xmax] = obj_class if obj_class is not None else obj_annotation.cls

    def get_map(self):
        return self._map.long()


class ClsnonmutexGenerator(GeneratorIf):
    """GT map class generator."""
    @property
    def fill_value(self):
        return 0.0  # Background class

    def _get_num_maps(self):
        return len(self._class_map.get_ids())

    def add_obj(self, obj_annotation, map_coords):
        xmin, ymin, xmax, ymax = map_coords

        if hasattr(obj_annotation, 'keypoint_detectability'):
            keypoint_detectability_ds = self._downsample_keypoint_detectability(obj_annotation.keypoint_detectability)
            mask = keypoint_detectability_ds > 0
            self._map[obj_annotation.cls-2, ymin: ymax, xmin: xmax][mask] = keypoint_detectability_ds[mask]
        else:
            self._map[obj_annotation.cls-2, ymin: ymax, xmin: xmax] = 1.0

    def get_map(self):
        return self._map


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
        if hasattr(obj_annotation, 'corners'):
            corner_coords = obj_annotation.corners
        else:
            rotation = matrix_from_yaw(obj_annotation.rot_y) if hasattr(obj_annotation, 'rot_y') \
                       else obj_annotation.rotation
            corner_coords = project_3d_pts(construct_3d_box(obj_annotation.size),
                                           self._calib,
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


class KeypointsGenerator(GeneratorIndex):
    """GT map KeypointsGenerator."""
    def _get_num_maps(self):
        return 2*NBR_KEYPOINTS

    def add_obj(self, obj_annotation, map_coords):
        xmin, ymin, xmax, ymax = map_coords
        rotation = matrix_from_yaw(obj_annotation.rot_y) if hasattr(obj_annotation, 'rot_y') \
                   else obj_annotation.rotation
        obj_label = self._class_map.label_from_id(obj_annotation.cls)
        assert self._get_num_maps() % 2 == 0
        keypoints_3d = self._metadata['objects'][obj_label]['keypoints']
        assert keypoints_3d.shape[1] == NBR_KEYPOINTS
        keypoints_2d = project_3d_pts(
            keypoints_3d,
            self._calib,
            obj_annotation.location,
            rot_matrix=rotation,
        )
        for index in range(NBR_KEYPOINTS):
            x, y = keypoints_2d[:, index]
            # Quite arbitrary threshold: 10
            # If keypoint is slightly outside of image is OK
            # If keypoint is far off, it will hurt training
            if abs(x / self._configs.data.img_dims[1] - 0.5) > 10:
                print('Keypoint projected far outside image. Check object not behind camera.')
            self._map[2 * index + 0, ymin: ymax, xmin: xmax] = x - self._index_map[1, ymin: ymax, xmin: xmax]
            self._map[2 * index + 1, ymin: ymax, xmin: xmax] = y - self._index_map[0, ymin: ymax, xmin: xmax]


class KeypointGenerator(GeneratorIndex):
    """GT map KeypointGenerator."""
    def _get_num_maps(self):
        return 2

    def add_obj(self, obj_annotation, map_coords):
        xmin, ymin, xmax, ymax = map_coords
        x, y = obj_annotation.keypoint

        if hasattr(obj_annotation, 'keypoint_detectability'):
            keypoint_detectability_ds = self._downsample_keypoint_detectability(obj_annotation.keypoint_detectability)
            mask = keypoint_detectability_ds > 0
            self._map[0, ymin: ymax, xmin: xmax][mask] = (x - self._index_map[1, ymin: ymax, xmin: xmax])[mask]
            self._map[1, ymin: ymax, xmin: xmax][mask] = (y - self._index_map[0, ymin: ymax, xmin: xmax])[mask]
        else:
            self._map[0, ymin: ymax, xmin: xmax] = x - self._index_map[1, ymin: ymax, xmin: xmax]
            self._map[1, ymin: ymax, xmin: xmax] = y - self._index_map[0, ymin: ymax, xmin: xmax]
