"""Save results to disk."""
import os
import json
import numpy as np

from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')  # Overriding nuscenes backend

from nuscenes.eval.eval_utils import category_to_detection_name
from nuscenes.eval.nuscenes_eval import NuScenesEval
from nuscenes.utils.data_classes import Box

class ResultSaver:
    """ResultSaver."""
    def __init__(self, configs):
        self._configs = configs
        self._epoch_results = None

        self._result_dir = os.path.join(configs.experiment_path, 'detections')
        os.makedirs(self._result_dir, exist_ok=True)

        if configs.data.dataformat == 'nuscenes':
            self._nusc = configs.nusc
            self._init_results()

    def _init_results(self):
        self._epoch_results = {sample['token']: [] for sample in self._nusc.sample}

    def save(self, detections, mode):
        if self._configs.logging.save_nuscenes_format:
            for data_token, frame_detections in detections.items():
                self.save_nuscenes_format(data_token, frame_detections)
        if self._configs.logging.save_kitti_format:
            for frame_id, frame_detections in detections.items():
                self.save_frame_kitti(frame_id, frame_detections, mode)

    def summarize_epoch(self, mode):
        if self._configs.data.dataformat != 'nuscenes':
            return None  # TODO: implement evaluation for other datasets
        result_path = self.write_to_file()
        eval_set = 'teaser_' + mode
        if mode == 'test':  # TODO: until full dataset released
            eval_set = 'teaser_val'
        output_dir = os.path.join(self._result_dir, 'nuscenes_eval')
        nusc_eval = NuScenesEval(nusc=self._nusc,
                                 result_path=result_path,
                                 eval_set=eval_set,
                                 output_dir=output_dir)
        all_metrics = nusc_eval.run_eval()
        score = all_metrics[self._configs.evaluation.score]
        return score

    def write_to_file(self):
        result_path = os.path.join(self._result_dir, 'nuscenes_results.json')
        with open(result_path, 'w') as file:
            json.dump(self._epoch_results, file, indent=4)
        self._init_results()
        return result_path

    def save_nuscenes_format(self, data_token, frame_detections):
        # TODO: Make sure that not multiple sample_datas write to the same sample.
        sample_results = []
        sample_data = self._nusc.get('sample_data', data_token)
        sample_token = sample_data['sample_token']

        for detection in frame_detections:
            if min(detection['size']) < 0:
                print('Negative size. Applying `abs()`')
            location, rotation = self.get_nusc_global_pose(detection, sample_data)
            sample_result = {
                "sample_token": sample_token,
                "translation": location.tolist(),
                "size": abs(detection['size']).tolist(),
                "rotation": rotation.normalised.elements.tolist(),
                "velocity": 3 * [float('nan')],
                "detection_name": category_to_detection_name(detection['cls']),
                "detection_score": float(detection['confidence']),
                "attribute_scores": 8 * [-1]
            }
            sample_results.append(sample_result)
        self._epoch_results[sample_token] = sample_results

    def get_nusc_global_pose(self, box, sample_data):
        box = Box(center=box['location'],
                  size=box['size'][[1, 2, 0]],  # HWL to WLH
                  orientation=Quaternion(axis=(0, 0, 1), angle=box['rotation_y']))

        pose_record = self._nusc.get('ego_pose', sample_data['ego_pose_token'])
        cs_record = self._nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

        # Move box to ego vehicle coord system.
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))

        # Move box to global coord system.
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))

        return box.center, box.orientation

    def save_frame_kitti(self, frame_id, frame_detections, mode):
        save_dir = os.path.join(self._result_dir, mode, 'kitti_format', 'data')
        os.makedirs(save_dir, exist_ok=True)

        lines_to_write = []
        for detection in frame_detections:
            alpha = detection.get('alpha', -10)
            if isinstance(alpha, np.ndarray) and alpha.shape == (2,):
                alpha = np.arctan2(*alpha)
            bbox2d = self._clip_bbox(detection.bbox2d)
            size = detection.get('size', [-1] * 3)
            location = detection.get('location', [-1000] * 3)
            write_line = ('{cls:} -1 -1 {alpha:=.5g} '
                          '{left:=.2f} {top:=.2f} {right:=.2f} {bottom:=.2f} '
                          '{height:=.3g} {width:=.3g} {length:=.3g} '
                          '{x:=.4g} {y:=.4g} {z:=.4g} {rotation_y:=.3g} '
                          '{score:=.6f}\n'.
                          format(cls=detection.cls,
                                 alpha=alpha,
                                 left=bbox2d[0],
                                 top=bbox2d[1],
                                 right=bbox2d[2],
                                 bottom=bbox2d[3],
                                 height=size[0],
                                 width=size[1],
                                 length=size[2],
                                 x=location[0],
                                 y=location[1] + size[0] / 2,  # Kitti format
                                 z=location[2],
                                 rotation_y=detection.get('rotation_y', -10),
                                 score=detection.confidence))
            if (np.array(size) > np.array([0.5, 0.2, 0.1])).all():
                lines_to_write.append(write_line)
            else:
                print('Warning, small or negative size: Skipping writing')
        with open(os.path.join(save_dir, '%06d.txt' % int(frame_id)), 'w') as file:
            file.writelines(lines_to_write)

    def _clip_bbox(self, bbox):
        return [max(0, bbox[0]),
                max(0, bbox[1]),
                min(self._configs.data.img_dims[1] - 1, bbox[2]),
                min(self._configs.data.img_dims[0] - 1, bbox[3])]
