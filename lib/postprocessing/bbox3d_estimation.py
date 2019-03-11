"""Estimate 3D bounding boxes for detected objects."""
from multiprocessing import Pool
import numpy as np
from scipy.optimize import least_squares
from lib.utils import project_3d_pts, construct_3d_box
from lib.utils import matrix_from_yaw
from lib.postprocessing import RunnerIf


class Runner(RunnerIf):
    def __init__(self, configs):
        super(Runner, self).__init__(configs, "bbox3d_estimation")

    def run(self, frame_detections, batch, frame_index):
        calibration = batch.calibration[frame_index]
        for detection in frame_detections:
            if not all(attr in detection for attr in ['corners', 'zdepth', 'size']):
                print("Estimation currently requires 'corners', 'zdepth' and 'size'")
                continue
            estimator = BoxEstimator(detection, calibration)
            box_parameters = estimator.heuristic_3d()
            if self._runner_configs.local_optimization_3d:
                box_parameters = estimator.solve()
            detection['size'] = box_parameters[:3]
            detection['location'] = box_parameters[3:6]
            detection['rotation_y'] = box_parameters[6]
            detection['rotation'] = matrix_from_yaw(box_parameters[6])
            detection['alpha'] = box_parameters[6] - np.arctan2(box_parameters[3], box_parameters[5])
        return frame_detections


class BoxEstimator:

    def __init__(self, data, calibration):
        self.data = data
        self._calibration = calibration

        self._weights = self._set_weights(data)
        self._solution = None

    def _set_weights(self, data):
        weights_list = []
        for modularity in ('corners', 'size', 'zdepth'):
            weights_list += list(data.get('w_' + modularity))
        return np.asarray(weights_list)

    def heuristic_3d(self):
        # Location
        obj_center = np.mean(self.data.corners, axis=1)
        obj_center = np.append(obj_center, 1)
        unprojection = np.linalg.pinv(self._calibration) @ obj_center
        location = unprojection[:3] * self.data.zdepth

        # Rotation
        corners_x = self.data.corners[0]
        south = sum(corners_x[[0, 3, 4, 7]] - corners_x[[1, 2, 5, 6]])  # left - right
        east = sum(corners_x[[0, 1, 2, 3]] - corners_x[[4, 5, 6, 7]])  # front - rear
        rotation = np.arctan2(south, east) + np.arctan2(location[0], location[2])

        return np.concatenate((self.data.size, location, [rotation]))

    def residuals(self, box_parameters):
        size = box_parameters[:3]
        location = box_parameters[3:6]
        rot_y = box_parameters[6]
        corners = project_3d_pts(construct_3d_box(size),
                                 self._calibration,
                                 location,
                                 rot_y=rot_y)

        residuals = np.concatenate(((self.data.corners - corners).flatten(),
                                    self.data.size - size,
                                    (self.data.zdepth - location[2],)))
        return self._weights * residuals

    def solve(self):
        self._solution = least_squares(fun=self.residuals,
                                       x0=self.heuristic_3d(),
                                       method='lm',
                                       xtol=1e-3)
        return self._solution.x
