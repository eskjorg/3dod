"""Save results to disk."""
import os
import numpy as np

class ResultSaver:
    """ResultSaver."""
    def __init__(self, configs):
        self._configs = configs

    def save(self, detections, mode):
        if self._configs.logging.save_kitti_format:
            for frame_id, frame_detections in detections.items():
                self.save_frame_kitti(frame_id, frame_detections, mode)

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
                                 y=location[1],
                                 z=location[2],
                                 rotation_y=detection.get('rotation', -10),
                                 score=detection.confidence))
            if (np.array(size) > np.array([0.5, 0.2, 0.1])).all():
                lines_to_write.append(write_line)
            else:
                print('Warning, negative size: Skipping writing')
        with open(os.path.join(save_dir, '{:06d}.txt'.format(int(frame_id))), 'w') as file:
            file.writelines(lines_to_write)

    def _clip_bbox(self, bbox):
        return [max(0, bbox[0]),
                max(0, bbox[1]),
                min(self._configs.data.img_dims[1] - 1, bbox[2]),
                min(self._configs.data.img_dims[0] - 1, bbox[3])]
