"""Save results to disk."""
import os
import json
import numpy as np

from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')  # Overriding nuscenes backend
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.eval_utils import category_to_detection_name
from nuscenes.utils.data_classes import Box

class ResultSaver:
    """ResultSaver."""
    def __init__(self, configs):
        self._configs = configs
        self._epoch_results = dict()
        if configs.data.dataformat == 'nuscenes':
            self._nusc = NuScenes(version='v0.1', dataroot=configs.data.path, verbose=True)

    def save(self, detections, mode):
        if self._configs.logging.save_nuscenes_format:
            for data_token, frame_detections in detections.items():
                self.save_nuscenes_format(data_token, frame_detections)
        if self._configs.logging.save_kitti_format:
            for frame_id, frame_detections in detections.items():
                self.save_frame_kitti(frame_id, frame_detections, mode)

    def save_nuscenes_format(self, data_token, frame_detections):
        # TODO: Make sure that not multiple sample_datas write to the same sample.
        sample_results = []
        sample_data = self._nusc.get('sample_data', data_token)
        sample_token = sample_data['sample_token']

        for detection in frame_detections:
            location, rotation = self.get_nusc_global_pose(detection, sample_data)
            sample_result = {
                "sample_token": sample_token,
                "translation": location,
                "size": detection['size'],
                "rotation": rotation,
                "velocity": 3 * [np.nan],
                "detection_name": category_to_detection_name(detection['cls']),
                "detection_score": detection['confidence'],
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
        save_dir = os.path.join(self._configs.experiment_path, 'detections', mode, 'kitti_format')
        os.makedirs(save_dir, exist_ok=True)

        lines_to_write = []
        for detection in frame_detections:
            bbox2d = self._clip_bbox(detection.bbox2d)
            size = detection.get('size', [-1] * 3)
            location = detection.get('location', [-1000] * 3)
            write_line = ('{cls:} -1 -1 {alpha:=.5g} '
                          '{left:=.2f} {top:=.2f} {right:=.2f} {bottom:=.2f} '
                          '{height:=.3g} {width:=.3g} {length:=.3g} '
                          '{x:=.4g} {y:=.4g} {z:=.4g} {rotation_y:=.3g} '
                          '{score:=.6f}\n'.
                          format(cls=detection.cls,
                                 alpha=detection.get('alpha', -10),
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
                print('Warning, negative size: Skipping writing')
        with open(os.path.join(save_dir, '%6d.txt' % int(frame_id)), 'w') as file:
            file.writelines(lines_to_write)

    def _clip_bbox(self, bbox):
        return [max(0, bbox[0]),
                max(0, bbox[1]),
                min(self._configs.data.img_dims[1] - 1, bbox[2]),
                min(self._configs.data.img_dims[0] - 1, bbox[3])]
