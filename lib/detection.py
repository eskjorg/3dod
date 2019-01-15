"""2D and 3D Detection from network outputs."""
from collections import defaultdict
from multiprocessing import Pool

import torch

from lib.data import maps

class Detector:
    """Detector."""
    def __init__(self, configs):
        super(Detector, self).__init__()
        self._configs = configs

    def run_detection(self, batch, outputs):
        """Run 2D and 3D detection."""
        batch_detections = defaultdict(dict)
        # for frame_index, frame_id in enumerate(batch.id):
        #     result = self._2d_detection(frame_index, outputs)
        #     result = self._3d_estimation(result, batch.calibration[frame_index])
        #     batch_detections.update({frame_id: result})
        print("Not yet implemented: ", 'Detector._3d_estimation()')
        return batch_detections

    def _2d_detection(self, frame_index, outputs):
        frame_outputs = {}
        for key in outputs:
            data = outputs[key][frame_index].to(device=torch.device('cpu')).detach().float()
            Generator = getattr(maps, key.capitalize() + 'Generator')
            frame_outputs[key] = Generator(self._configs).decode(data)
        frame_results = []
        #pool = Pool(25)
        for class_index in range(2, max(self._configs.data.class_map.values())):
            break
            map_split = split_maps(self._configs.inference.nms_split,
                                   (frame_outputs['class'][class_index],
                                    frame_outputs['bbox2d']))
            indices = pool.map(nms, map_split)
        return frame_results

    def _3d_estimation(self, result, calibration):
        return result
