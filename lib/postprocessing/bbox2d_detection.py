"""Run thresholding and nms."""
from attrdict import AttrDict

import torch
from maskrcnn_benchmark.layers import nms

from lib.constants import NBR_KEYPOINTS
from lib.data import maps
from lib.utils import get_device, get_class_map
from lib.postprocessing import RunnerIf

class Runner(RunnerIf):
    def __init__(self, configs):
        super(Runner, self).__init__(configs, "bbox2d_detection")
        self._class_map = get_class_map(configs)

    def run(self, outputs, batch, frame_index):
        frame_outputs = {}
        for key in outputs:
            data, weights = self.collect_data(outputs, key, frame_index)
            frame_outputs[key] = data
            frame_outputs['w_' + key] = weights
        frame_results = []
        for class_index in self._class_map.get_ids():
            frame_results += self.detect_class(frame_outputs, class_index)
        if len(frame_results) > 100:
            print('Frame %02d:' % frame_index, "More than 100 detections, skipping frame")
            frame_results = []
        return frame_results

    def collect_data(self, outputs, key, frame_index):
        # data
        data = outputs[key][0][frame_index].detach()
        Generator = getattr(maps, key.capitalize() + 'Generator')
        metadata = None
        data = Generator(self._configs, metadata, self._class_map, device=get_device()).decode(data)
        data = data.permute(1, 2, 0).reshape(-1, data.shape[0]).float()
        # weights
        weighting_mode = self._configs.training.weighting_mode
        if key == "cls" or weighting_mode == 'uniform':
            weight = getattr(self._configs.postprocessing.bbox3d_estimation.weights, key, 0)
            weights = torch.full(size=data.shape, fill_value=weight)
        elif weighting_mode == "layer_wise":
            weights = (-outputs[key][1].detach()).exp().expand(data.shape)
        elif weighting_mode == 'sample_wise':
            weights = (-outputs[key][1][frame_index].detach()).exp()
            weights = weights.permute(1, 2, 0).reshape(-1, weights.shape[0]).float()
        return data, weights

    def detect_class(self, frame_outputs, class_index):
        confidence_vector = frame_outputs['cls'][:, class_index]
        indices = torch.arange(len(confidence_vector))
        confident = indices[confidence_vector >= self._runner_configs.detection_threshold]
        if len(confident) == 0:
            return []
        indices = nms(frame_outputs['bbox2d'][confident],
                      confidence_vector[confident],
                      self._runner_configs.iou_threshold)
        result_dict = {key: mask[confident][indices].to(torch.device('cpu')).numpy()
                       for key, mask in frame_outputs.items()}
        result_dict['confidence'] = result_dict['cls'][:, class_index]
        result_dict['cls'] = [self._class_map.label_from_id(class_index)] * len(indices)
        if 'zdepth' in result_dict:
            result_dict['zdepth'] = result_dict['zdepth'].squeeze(axis=1)
        if 'corners' in result_dict:
            result_dict['corners'] = result_dict['corners'].reshape(-1, 2, 8, order='F')
        if 'keypoints' in result_dict:
            result_dict['keypoints'] = result_dict['keypoints'].reshape(-1, 2, NBR_KEYPOINTS, order='F')
        return [AttrDict(zip(result_dict, detection)) for detection in zip(*result_dict.values())]
