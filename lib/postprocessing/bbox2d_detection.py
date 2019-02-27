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
        outputs_task, outputs_ln_b = outputs
        frame_outputs = {}
        for key in outputs_task:
            data = outputs_task[key][frame_index].detach()
            Generator = getattr(maps, key.capitalize() + 'Generator')
            data = Generator(self._configs, metadata=None, device=get_device()).decode(data)
            frame_outputs[key] = data.permute(1, 2, 0).reshape(-1, data.shape[0]).squeeze().float()
        for key in outputs_ln_b:
            data = outputs_ln_b[key][frame_index].detach()
            frame_outputs[key + '_ln_b'] = data.permute(1, 2, 0).reshape(-1, data.shape[0]).squeeze().float()
        frame_results = []
        for class_index in self._class_map.get_ids():
            confidence_vector = frame_outputs['cls'][:, class_index]
            indices = torch.arange(len(confidence_vector))
            confident = indices[confidence_vector >= self._runner_configs.detection_threshold]
            if len(confident):
                indices = nms(frame_outputs['bbox2d'][confident],
                              confidence_vector[confident],
                              self._runner_configs.iou_threshold)
                result_dict = {key: mask[confident][indices].to(torch.device('cpu')).numpy()
                               for key, mask in frame_outputs.items()}
                result_dict['confidence'] = result_dict['cls'][:, class_index]
                result_dict['cls'] = [self._class_map.label_from_id(class_index)] * len(indices)
                if 'corners' in result_dict:
                    result_dict['corners'] = result_dict['corners'].reshape(-1, 2, 8, order='F')
                if 'keypoints' in result_dict:
                    result_dict['keypoints'] = result_dict['keypoints'].reshape(-1, 2, NBR_KEYPOINTS, order='F')
                frame_results += [AttrDict(zip(result_dict, detection))
                                  for detection in zip(*result_dict.values())]
        if len(frame_results) > 100:
            print('Frame %02d:' % frame_index, "More than 100 detections, skipping frame")
            frame_results = []
        return frame_results
