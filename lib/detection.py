"""2D and 3D Detection from network outputs."""
from collections import defaultdict
from multiprocessing import Pool

import torch

from maskrcnn_benchmark.layers import nms

from lib.data import maps
from lib.estimation import BoxEstimator
from lib.utils import get_device


class Detector:
    """Detector."""
    def __init__(self, configs):
        super(Detector, self).__init__()
        self._configs = configs

    def run_detection(self, batch, outputs):
        """Run 2D and 3D detection."""
        batch_detections = defaultdict(dict)
        for frame_index, frame_id in enumerate(batch.id):
            result = self._2d_detection(frame_index, outputs)
            result = self._3d_estimation(result, batch.calibration[frame_index])
            batch_detections.update({frame_id: result})
        return batch_detections

    def _2d_detection(self, frame_index, outputs):
        frame_outputs = {}
        for key in outputs:
            data = outputs[key][frame_index].detach()
            Generator = getattr(maps, key.capitalize() + 'Generator')
            data = Generator(self._configs, device=get_device()).decode(data)
            frame_outputs[key] = data.permute(1, 2, 0).reshape(-1, data.shape[0]).float()
        frame_results = {}
        for class_index in range(2, max(self._configs.data.class_map.values())):
            confidence_vector = frame_outputs['class'][:, class_index]
            indices = torch.arange(len(confidence_vector))
            confident = indices[confidence_vector >= self._configs.detection.detection_threshold]
            if len(confident):
                indices = nms(frame_outputs['bbox2d'][confident],
                              confidence_vector[confident],
                              self._configs.detection.iou_threshold)
                result_dict = {key: mask[confident][indices].to(torch.device('cpu')).numpy()
                               for key, mask in frame_outputs.items()}
                frame_results[class_index] = [dict(zip(result_dict, detection)) for detection in zip(*result_dict.values())]
        return frame_results

    def _3d_estimation(self, frame_detections, calibration):
        for class_detections in frame_detections.values():
            if len(class_detections) > 100:
                print("More than 100 detections: Skipping 3D")
                continue
            for detection in class_detections:
                estimator = BoxEstimator(detection, calibration, self._configs.estimation.weights)
                box_parameters = estimator.heuristic_3d()
                if self._configs.estimation.local_optimization_3d:
                    box_parameters = estimator.solve()
                detection['size'] = box_parameters[:3]
                detection['location'] = box_parameters[3:6]
                detection['rotation'] = box_parameters[6]
        return frame_detections
