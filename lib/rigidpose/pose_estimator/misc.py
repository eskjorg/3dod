import os
import json
import math
import numpy as np
from lib.rigidpose.pose_estimator.gencode import *
import pyopengv

def invert_eucl(eucl):
    return np.concatenate([eucl[:,:3].T, -eucl[:,:3].T@eucl[:,[3]]], axis=1)

def p3p_kneip(u, U):
    viewing_rays = u / np.linalg.norm(u, axis=0)
    cameras = pyopengv.absolute_pose_p3p_kneip(viewing_rays.T, U.T)
    cameras = [invert_eucl(cam) for cam in cameras if np.all(np.isfinite(cam))]
    return cameras

def opengv_ransac(u, U):
    viewing_rays = u / np.linalg.norm(u, axis=0)
    method_name = 'KNEIP' # This is the default
    # method_name = 'EPNP'
    cam = pyopengv.absolute_pose_ransac(viewing_rays.T, U.T, method_name, 0.02, iterations=1000, probability=0.99)
    cam = invert_eucl(cam)
    return cam if np.all(np.isfinite(cam)) else None

def epnp(u, U):
    viewing_rays = u / np.linalg.norm(u, axis=0)
    cam = pyopengv.absolute_pose_epnp(viewing_rays.T, U.T)
    cam = invert_eucl(cam)
    return cam if np.all(np.isfinite(cam)) else None

def normalize(u_unnorm, K):
    return np.dot(np.linalg.inv(K), u_unnorm)

def denormalize(u_norm, K):
    return np.dot(K, u_norm)

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

def deg_cm_error(R_pred, t_pred, R_gt, t_gt, rescale2meter_factor=1e-3):
    cos_theta_est = (np.trace(np.dot(R_pred.T, R_gt)) - 1) / 2.
    cos_theta = np.clip(cos_theta_est, -1., 1.)
    assert abs(cos_theta - cos_theta_est) < 1e-2
    deg_error = 180./np.pi * np.arccos(cos_theta)
    cm_error = 100. * rescale2meter_factor * np.linalg.norm(t_pred - t_gt)
    return deg_error, cm_error
