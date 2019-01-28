"""2D and 3D Detection from network outputs."""
from collections import defaultdict
from multiprocessing import Pool

from attrdict import AttrDict

import numpy as np
import torch

from maskrcnn_benchmark.layers import nms

from lib.data import maps
from lib.estimation import BoxEstimator
from lib.utils import get_device, get_class_map


class Detector:
    """Detector."""
    def __init__(self, configs):
        super(Detector, self).__init__()
        self._configs = configs
        self._class_map = get_class_map(configs)

    def run_detection(self, batch, outputs):
        """Run 2D and 3D detection."""
        batch_detections = defaultdict(dict)
        for frame_index, frame_id in enumerate(batch.id):
            result = self._2d_detection(frame_index, outputs)
            result = self._3d_estimation(result, batch.calibration[frame_index])
            batch_detections.update({frame_id: result})
        return batch_detections

    def _2d_detection(self, frame_index, outputs):
        outputs_task, outputs_ln_b = outputs
        frame_outputs = {}
        for key in outputs_task:
            data = outputs_task[key][frame_index].detach()
            Generator = getattr(maps, key.capitalize() + 'Generator')
            data = Generator(self._configs, device=get_device()).decode(data)
            frame_outputs[key] = data.permute(1, 2, 0).reshape(-1, data.shape[0]).squeeze().float()
        for key in outputs_ln_b:
            data = outputs_ln_b[key][frame_index].detach()
            frame_outputs[key + '_ln_b'] = data.permute(1, 2, 0).reshape(-1, data.shape[0]).squeeze().float()
        frame_results = []
        for class_index in self._class_map.get_ids():
            confidence_vector = frame_outputs['cls'][:, class_index]
            indices = torch.arange(len(confidence_vector))
            confident = indices[confidence_vector >= self._configs.detection.detection_threshold]
            if len(confident):
                indices = nms(frame_outputs['bbox2d'][confident],
                              confidence_vector[confident],
                              self._configs.detection.iou_threshold)
                result_dict = {key: mask[confident][indices].to(torch.device('cpu')).numpy()
                               for key, mask in frame_outputs.items()}
                result_dict['confidence'] = result_dict['cls'][:, class_index]
                result_dict['cls'] = [self._class_map.label_from_id(class_index)] * len(indices)
                if 'corners' in result_dict:
                    result_dict['corners'] = result_dict['corners'].reshape(-1, 2, 8, order='F')
                frame_results += [AttrDict(zip(result_dict, detection))
                                  for detection in zip(*result_dict.values())]
        return frame_results

    def _3d_estimation(self, frame_detections, calibration):
        if len(frame_detections) > 100:
            print("More than 100 detections: Skipping 3D")
        else:
            for detection in frame_detections:
                if not all(attr in detection for attr in ['corners', 'zdepth', 'size']):
                    print("Estimation currently requires 'corners', 'zdepth' and 'size'")
                    continue
                if self._configs.training.nll_loss:
                    weights = self._configs.estimation.weights  # TODO: Optimize with L1 loss
                else:
                    weights = self._configs.estimation.weights
                estimator = BoxEstimator(detection, calibration, weights)
                box_parameters = estimator.heuristic_3d()
                if self._configs.estimation.local_optimization_3d:
                    box_parameters = estimator.solve()
                detection['size'] = box_parameters[:3]
                detection['location'] = box_parameters[3:6]
                detection['rotation'] = box_parameters[6]
                detection['alpha'] = box_parameters[6] - np.arctan2(box_parameters[3], box_parameters[5])
        return frame_detections
