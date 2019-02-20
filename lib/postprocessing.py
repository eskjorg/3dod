"""Estimate 3D bounding boxes for detected objects."""
import numpy as np
from scipy.optimize import least_squares
from lib.utils import project_3d_pts, construct_3d_box


class BoxEstimator:

    def __init__(self, data, calibration, weights):
        self.data = data
        self._calibration = calibration

        self._weights = self._set_weights(weights)
        self._solution = None

    def _set_weights(self, weights_dict):
        weights_list = []
        for modality, data in weights_dict.items():
            weights_list += self.data[modality].size * [data]
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
        corners = project_3d_pts(
            construct_3d_box(size),
            self._calibration,
            location,
            rot_y=rot_y,
        )

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
