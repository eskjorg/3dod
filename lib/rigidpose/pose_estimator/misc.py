import os
import json
import math
import numpy as np
from pose_estimator.gencode import *

with open(os.path.join('..', 'local_conf.json'), 'r') as f:
    local_conf = json.load(f)
with open(os.path.join('..', 'meta.json'), 'r') as f:
    meta = json.load(f)

def get_camera_calibration_matrix():
    K = np.array([
        [meta['camera_calibration']['f_x'], 0.0                              , meta['camera_calibration']['p_x']],
        [                              0.0, meta['camera_calibration']['f_y'], meta['camera_calibration']['p_y']],
        [                              0.0,                               0.0,                               1.0],
    ])
    return K

def normalize(u_unnorm):
    K = get_camera_calibration_matrix()
    return np.dot(np.linalg.inv(K), u_unnorm)

def denormalize(u_norm):
    K = get_camera_calibration_matrix()
    return np.dot(K, u_norm)

def xyz_read(filename):
    with open(filename) as f:
        xyz_rows = np.array([[float(token) for token in row.strip().split()] for row in f])
    assert len(xyz_rows.shape) == 2
    assert xyz_rows.shape[1] == 3
    xyz_cols = xyz_rows.T
    return xyz_cols

def get_mesh_dict():
    meshes = {};
    for obj in meta['objects']:
        if meta['objects'][obj]['mesh_id'] is not None:
            # Mesh exists for object
            meshes[obj] = xyz_read(os.path.join(local_conf['xyz_models_path'], '{:03}.xyz'.format(meta['objects'][obj]['mesh_id'])))
    return meshes

def pflat(x):
    a, n = x.shape
    alpha = x[np.newaxis,-1,:]
    return x / alpha

def pextend(x):
    sx, sy = x.shape
    return np.concatenate([x, np.ones((1,sy))], axis=0)

def getcam(x):
    assert x.shape == (1,6)
    rx, ry, rz = x[0,0:3]
    R = np.array([
        [math.cos(ry)*math.cos(rz),                                             -math.cos(ry)*math.sin(rz),                                               math.sin(ry)             ],
        [math.cos(rx)*math.sin(rz) + math.cos(rz)*math.sin(rx)*math.sin(ry),     math.cos(rx)*math.cos(rz) - math.sin(rx)*math.sin(ry)*math.sin(rz),     -math.cos(ry)*math.sin(rx)],
        [math.sin(rx)*math.sin(rz) - math.cos(rx)*math.cos(rz)*math.sin(ry),     math.cos(rz)*math.sin(rx) + math.cos(rx)*math.sin(ry)*math.sin(rz),      math.cos(rx)*math.cos(ry)],
    ])
    P = np.concatenate([R, -np.dot(R, x[np.newaxis,0,3:6].T)], axis=1);
    return P

def calc_res(x, U, um):
    assert x.shape == (1,6)
    assert U.shape[0] == 4
    assert um.shape[0] == 2
    P = getcam(x)
    u = pflat(np.dot(P, U))
    return np.reshape(u[0:2,:]-um, (-1,), order='F')

def poseobjective(x, U, um, alpha_rho, c, w=None):
    assert x.shape == (1,6)
    assert U.shape[0] == 4
    assert um.shape[0] == 2
    res = calc_res(x, U, um)
    rho = calc_rho_grad(res, alpha_rho, c, w=w)
    fval = np.sum(rho['rho'])
    dres_dx = calc_dres_dx(x, U)
    drho_dx = np.sum(np.tile(rho['drho_dres'][:,np.newaxis], (1,6))*dres_dx, axis=0)
    return fval, drho_dx

def deg_cm_error(R_pred, t_pred, R_gt, t_gt):
    cos_theta_est = (np.trace(np.dot(R_pred.T, R_gt)) - 1) / 2.
    cos_theta = np.clip(cos_theta_est, -1., 1.)
    assert abs(cos_theta - cos_theta_est) < 1e-2
    deg_error = 180./np.pi * np.arccos(cos_theta)
    cm_error = 100. * np.linalg.norm(t_pred - t_gt)
    return deg_error, cm_error
