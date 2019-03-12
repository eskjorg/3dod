import os
import shutil
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

import time

from utils import write_pose
from pose_estimator.visionary_resection import resec3pts
from pose_estimator.gencode import *
from pose_estimator.misc import *

with open(os.path.join('..', 'meta.json'), 'r') as f:
    meta = json.load(f)

mesh_dict = get_mesh_dict()

clock_flag = False

class PoseEstimator():
    def __init__(
            self,
            U0,
            u_unnorm,
            obj,
            verbose=0,
            w=None,
            alpha_rho=-1.0,
            c=0.002,
            nransac=500,
            ransacthr=0.01,
            lambda0=1e5,
            max_refine_iters=200,
        ):
        assert len(u_unnorm.shape) == 2
        assert len(U0.shape) == 2
        assert u_unnorm.shape[0] == 2
        assert U0.shape[0] == 3

        assert np.all(np.isfinite(U0))

        self.ntotal = U0.shape[1]
        if not self.ntotal >= 3:
            print(U0)
            print(u_unnorm)
        assert self.ntotal >= 3
        assert u_unnorm.shape[1] == self.ntotal
        if w is None:
            w = np.ones((self.ntotal,))
        assert w.shape == (self.ntotal,)

        self.verbose = verbose

        self.U0 = pextend(U0)
        self.u_unnorm = u_unnorm
        self.um = normalize(pextend(u_unnorm))[0:2,:]
        self.obj = obj
        self.w = w

        self.alpha_rho = alpha_rho
        self.c = c
        self.nransac = nransac
        self.ransacthr = ransacthr
        self.lambda0 = lambda0
        self.max_refine_iters = max_refine_iters

        # Mesh vertices
        self.Uanno0 = pextend(mesh_dict[obj])

    def pose_forward(self):
        # INITIAL RANSAC ESTIMATE
        if clock_flag:
            t0 = time.time()
        P0, nbest = ransac(self.U0, self.um, self.nransac, self.ransacthr, verbose=self.verbose)
        if clock_flag:
            print("{:>15} time: {}".format("RANSAC", time.time()-t0))

        # Transformation matrix T0. Choosing scale based on data.
        tmp = np.dot(P0, self.U0)
        self.sc = 1. / np.std(tmp)
        self.T0 = np.concatenate([self.sc*P0, np.array([[0,0,0,1]])], axis=0)

        # Transform points using initial estimate
        self.U = np.dot(self.T0, self.U0)
        self.Uanno = np.dot(self.T0, self.Uanno0)

        # LOCAL REFINEMENT
        if clock_flag:
            t0 = time.time()
        x0 = np.zeros((1,6)) # Since points are transformed
        self.x, self.drhototal_dx, self.rhototal, self.refine_iters, final_lambda_refinement = local_refinement(x0, self.U, self.um, self.w, self.alpha_rho, self.c, self.max_refine_iters, self.lambda0, verbose=self.verbose)
        self.final_res = np.reshape(calc_res(self.x, self.U, self.um), (2,-1), order='F')
        if clock_flag:
            print("{:>15} time: {}".format("refinement", time.time()-t0))
        if self.verbose >= 1:
            print("drhototal_dx: {}".format(self.drhototal_dx))
            print("drhototal_dx_norm: {}".format(np.linalg.norm(self.drhototal_dx)))
        if np.linalg.norm(self.drhototal_dx) > 1e-1:
            raise BigGradientException()

        self.P_pred = np.dot(getcam(self.x), self.T0) / self.sc
        R_pred = self.P_pred[:,0:3]
        t_pred = self.P_pred[:,3,np.newaxis]
        return R_pred, t_pred, self.rhototal, self.drhototal_dx, self.final_res, self.refine_iters, final_lambda_refinement

    def pose_backward(self, Ranno, tanno):
        # Ground truth projection of mesh vertices
        self.Panno = np.concatenate([Ranno, tanno], axis=1)
        self.umanno = pflat(np.dot(self.Panno, self.Uanno0))[0:2,:]

        # COMPUTE LOSS (reprojection error for mesh vertices)
        if clock_flag:
            t0 = time.time()
        resanno = calc_res(self.x, self.Uanno, self.umanno)
        self.loss = 0.5*np.dot(resanno.T, resanno)
        if clock_flag:
            print("{:>15} time: {}".format("loss", time.time()-t0))

        # COMPUTE DERIVATIVES
        if clock_flag:
            t0 = time.time()
        self.dloss_dU, self.dloss_dw = compute_derivatives(self.Uanno, self.umanno, self.U, self.um, self.w, self.x, self.alpha_rho, self.c, verbose=self.verbose)
        if clock_flag:
            print("{:>15} time: {}".format("derivatives", time.time()-t0))

        mean_pixel_reproj_err = 0.5*(meta['camera_calibration']['f_x']+meta['camera_calibration']['f_y']) * np.mean(np.sqrt(np.sum(np.reshape(resanno, (2,-1), order='F')**2, axis=0)))

        # Gradient step should be independent of number of mesh vertices
        self.loss /= len(resanno)
        self.dloss_dU /= len(resanno)
        self.dloss_dw /= len(resanno)

        return self.dloss_dU, self.dloss_dw, self.loss, mean_pixel_reproj_err

    def plot(self, prefix=''):
        from matplotlib import pyplot as plt
        def scatter(u, fname):
            plt.figure()
            plt.scatter(u[0,:], 480-u[1,:])
            plt.xlim((0, 640))
            plt.ylim((0, 480))
            plt.savefig(os.path.join(outdir, fname))
        outdir = '../evaluation/out/tmp'
        os.makedirs(outdir, exist_ok=True)

        write_pose(os.path.join(outdir, '{}{}_Panno.txt'.format(prefix, self.obj)), self.Panno[:,0:3], self.Panno[:,3,np.newaxis])
        write_pose(os.path.join(outdir, '{}{}_P_pred.txt'.format(prefix, self.obj)), self.P_pred[:,0:3], self.P_pred[:,3,np.newaxis])

        # # Do correspondences map to each other?
        # for idx in [0, self.ntotal//2, self.ntotal-1]:
        #     scatter(self.u_unnorm[:,idx,np.newaxis], '{}_{}_2d.png'.format(self.obj, idx))
        #     scatter(denormalize(pflat(np.dot(self.Panno, self.U0)))[0:2,idx,np.newaxis], '{}_{}_reproj.png'.format(self.obj, idx))

        scatter(self.u_unnorm, '{}{}_2d.png'.format(prefix, self.obj))
        # scatter(denormalize(pextend(self.um))[0:2, :], '{}{}_2d_denorm.png'.format(prefix, self.obj))
        scatter(denormalize(pflat(np.dot(self.Panno, self.U0)))[0:2,:], '{}{}_reproj_obs_anno.png'.format(prefix, self.obj))
        scatter(denormalize(pflat(np.dot(self.P_pred, self.U0)))[0:2,:], '{}{}_reproj_obs_pred.png'.format(prefix, self.obj))
        scatter(denormalize(pflat(np.dot(self.Panno, self.Uanno0)))[0:2,:], '{}{}_reproj_mesh_anno.png'.format(prefix, self.obj))
        scatter(denormalize(pflat(np.dot(self.P_pred, self.Uanno0)))[0:2,:], '{}{}_reproj_mesh_pred.png'.format(prefix, self.obj))
        # assert False

def draw_3d_bounding_box(out_path, img, poses_gt, poses_est, correct_flags, fig_transform=None):
    assert set(poses_est.keys()) == set(correct_flags.keys())

    K = get_camera_calibration_matrix()

    # Save to file instead of interactive plotting
    # plt.ioff() # Not needed if using Agg backend (set above), since that does not support interactive pltting anyway

    # No frame, according to https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    fig = plt.figure(frameon=False)
    dpi = 100
    fig.set_size_inches(640/dpi, 480/dpi)

    # Make content fill whole figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img)

    # Don't rescale, even if bounding boxes are projected outside image
    ax.autoscale(enable=False)

    for obj in set(poses_gt.keys()) | set(poses_est.keys()):
        bounds = meta['objects'][obj]['bounds']
        corners = np.zeros((3, 8))

        corners[:,0] = [bounds['x'][1], bounds['y'][1], bounds['z'][1]]
        corners[:,1] = [bounds['x'][0], bounds['y'][1], bounds['z'][1]]
        corners[:,2] = [bounds['x'][0], bounds['y'][0], bounds['z'][1]]
        corners[:,3] = [bounds['x'][1], bounds['y'][0], bounds['z'][1]]
        corners[:,4] = [bounds['x'][1], bounds['y'][1], bounds['z'][0]]
        corners[:,5] = [bounds['x'][0], bounds['y'][1], bounds['z'][0]]
        corners[:,6] = [bounds['x'][0], bounds['y'][0], bounds['z'][0]]
        corners[:,7] = [bounds['x'][1], bounds['y'][0], bounds['z'][0]]

        corner_colors = ['k', 'w', 'r', 'g', 'y', 'c', 'm', 'b']

        # Now, project this into the image using the ground truth pose
        if obj in poses_gt:
            U = pflat(K @ (poses_gt[obj]['R'] @ corners + poses_gt[obj]['t']))
            xs = np.array([[U[0,0], U[0, 1], U[0, 2], U[0,3], U[0,4], U[0,5], U[0,6], U[0,7], U[0,0], U[0,1], U[0,2], U[0,3]], [U[0,1], U[0, 2], U[0,3], U[0,0], U[0,5], U[0,6], U[0,7], U[0,4], U[0,4], U[0,5], U[0,6], U[0,7]]])
            ys = np.array([[U[1,0], U[1,1], U[1,2], U[1,3], U[1,4], U[1,5], U[1,6], U[1,7], U[1,0], U[1,1], U[1,2], U[1,3]], [U[1,1], U[1,2], U[1,3], U[1,0], U[1,5], U[1,6], U[1,7], U[1,4], U[1,4], U[1,5], U[1,6], U[1,7]]])
            ax.plot([xs[0,:], xs[1,:]], [ys[0,:], ys[1,:]], color='b', linestyle='-', linewidth=1)
            corner_plot_args = []
            for pt, col in zip(U.T, corner_colors):
                corner_plot_args += [pt[0], pt[1], col+'.']
            ax.plot(*corner_plot_args)

        # Now, plot the estimated poses
        if obj in poses_est:
            U = pflat(K @ (poses_est[obj]['R'] @ corners + poses_est[obj]['t']))
            xs = np.array([[U[0,0], U[0, 1], U[0, 2], U[0,3], U[0,4], U[0,5], U[0,6], U[0,7], U[0,0], U[0,1], U[0,2], U[0,3]], [U[0,1], U[0, 2], U[0,3], U[0,0], U[0,5], U[0,6], U[0,7], U[0,4], U[0,4], U[0,5], U[0,6], U[0,7]]])
            ys = np.array([[U[1,0], U[1,1], U[1,2], U[1,3], U[1,4], U[1,5], U[1,6], U[1,7], U[1,0], U[1,1], U[1,2], U[1,3]], [U[1,1], U[1,2], U[1,3], U[1,0], U[1,5], U[1,6], U[1,7], U[1,4], U[1,4], U[1,5], U[1,6], U[1,7]]])
            ax.plot([xs[0,:], xs[1,:]], [ys[0,:], ys[1,:]], color='g' if correct_flags[obj] else 'r', linestyle=':', linewidth=1)
            corner_plot_args = []
            for pt, col in zip(U.T, corner_colors):
                corner_plot_args += [pt[0], pt[1], col+'.']
            ax.plot(*corner_plot_args)

    # Save to file instead of interactive plotting
    fig.savefig(out_path, dpi=dpi)
    if fig_transform is not None:
        fig_pil = Image.open(out_path).convert('RGB')
        fig_pil = fig_transform(fig_pil)
        fig_pil.save(out_path)
    plt.close(fig)

class RANSACException(Exception):
    pass

def ransac(U0, um, nransac, ransacthr, verbose=0):
    nbest = 0
    ntotal = um.shape[1]
    for i in range(nransac):
        ind = np.random.permutation(ntotal)
        ind = ind[0:3]

        cameras = resec3pts(pextend(um[:,ind]), U0[:,ind], coord_change=True)
        for P in cameras:
            u = pflat(np.dot(P, U0))
            res = np.sum((u[0:2,:] - um)**2, axis=0)
            ninliers = np.sum(res < ransacthr**2)
            if ninliers > nbest:
                Pransac = P
                nbest = ninliers

    if not nbest > 0:
        raise RANSACException()
    return Pransac, nbest

class BigGradientException(Exception):
    pass

def local_refinement(x, U, um, w, alpha_rho, c, niter, lambda0, verbose=0):
    """
    Local minimization of rho(res(x)) w.r.t. x, using Gauss-Newton
    """
    # iterative local optimization
    lambd = lambda0
    fval,dx = poseobjective(x,U,um,alpha_rho,c,w=w)
    iter = 0
    while iter<niter:
        iter += 1
        res = calc_res(x,U,um)
        dres_dx = calc_dres_dx(x,U)
        ddx_dres_dx = calc_ddx_dres_dx(x,U)
        rhoout = calc_rho_grad_hess(res,alpha_rho,c,w=w)

        # Rename derivatives to legacy names
        J = dres_dx
        H = ddx_dres_dx
        rhores,drdx,drda,drdc,dr2dx2,dr2dxda,dr2dxdc = rhoout['rho'],rhoout['drho_dres'],rhoout['drho_da'],rhoout['drho_dc'],rhoout['drho2_dres2'],rhoout['drho2_dresda'],rhoout['drho2_dresdc']

        dx = np.sum(np.tile(drdx[:,np.newaxis], (1,6)) * J, axis=0, keepdims=True)
        A = np.zeros((6,6))
        for k in range(6):
            for l in range(6):
                indmax = max(k,l)
                indmin = min(k,l)
                A[k,l] = np.sum(dr2dx2*J[:,k]*J[:,l] + drdx*H[:,indmin,indmax])

        dd = -np.dot(dx, np.linalg.inv(A+lambd*np.eye(6)).T)
        fvalnew,dxnew = poseobjective(x+dd,U,um,alpha_rho,c,w=w)
        if fvalnew < fval:
            dx = dxnew
            x += dd
            fval = fvalnew
            lambd *= 0.9
        else:
            lambd *= 2
        if np.linalg.norm(dx) < 1e-3:
            break

    if verbose >= 1:
        print("lambda: {}".format(lambd))
    return x, dx, fval, iter, lambd

def compute_derivatives(Uanno, umanno, U, um, w, x, alpha_rho, c, verbose=0):
    ntotal = um.shape[1]

    resanno = calc_res(x,Uanno,umanno)
    res = calc_res(x,U,um)

    dres_dx_anno = calc_dres_dx(x,Uanno)
    dres_dx = calc_dres_dx(x,U)
    ddx_dres_dx = calc_ddx_dres_dx(x,U)
    dres_dU, ddU_dres_dx = calc_ddU_dres_dx(x,U)
    rhoout = calc_rho_grad_hess(res,alpha_rho,c,w=w)

    # Rename derivatives to legacy names
    Janno = dres_dx_anno
    J = dres_dx
    H = ddx_dres_dx
    JU, HU = dres_dU, ddU_dres_dx
    rhores,drdx,drda,drdc,dr2dx2,dr2dxda,dr2dxdc = rhoout['rho'],rhoout['drho_dres'],rhoout['drho_da'],rhoout['drho_dc'],rhoout['drho2_dres2'],rhoout['drho2_dresda'],rhoout['drho2_dresdc']

    #solve for dxdU
    A = np.zeros((6,6))
    for k in range(6):
        for l in range(6):
            indmax = max(k,l)
            indmin = min(k,l)
            A[k,l] = np.sum(dr2dx2*J[:,k]*J[:,l] + drdx*H[:,indmin,indmax])
    Ainv = np.linalg.inv(A)

    drdx_without_w = drdx / np.kron(w, [1, 1])
    # drdx_without_w = drdx.copy()
    # drdx_without_w[::2] /= w
    # drdx_without_w[1::2] /= w

    dlossdU = np.zeros((3, ntotal))
    dlossdw = np.zeros(ntotal)

    dloss_dx = np.dot(resanno[np.newaxis,:],Janno)
    for jj in range(ntotal):
        bU = np.zeros((6,3))
        bw = np.zeros((6,1))
        for k in range(6):
            bU[k,:] = dr2dx2[2*jj]   * J[2*jj,k]   * JU[2*jj,:]        +    drdx[2*jj]   * HU[2*jj,k,:] +                           dr2dx2[2*jj+1] * J[2*jj+1,k] * JU[2*jj+1,:]      +    drdx[2*jj+1] * HU[2*jj+1,k,:]
            bw[k,0] = drdx_without_w[2*jj]*J[2*jj,k] + drdx_without_w[2*jj+1]*J[2*jj+1,k]
        dxdU = -np.dot(Ainv, bU)
        dxdw = -np.dot(Ainv, bw)

        dlossdU[:, jj] += np.squeeze(np.dot(dloss_dx, dxdU))
        dlossdw[jj] += np.squeeze(np.dot(dloss_dx, dxdw))
    dlossdU /= ntotal
    dlossdw /= ntotal
    return dlossdU, dlossdw

# def estimate_pose_test_interface(u_unnorm, U, obj):
#     assert len(u_unnorm.shape) == 2
#     assert len(U.shape) == 2
#     assert u_unnorm.shape[0] == 2
#     assert U.shape[0] == 3
#     assert u_unnorm.shape[1] == U.shape[1]
#     mesh = mesh_dict[obj]
#     first = False # Legacy
#     prefix = "" # Legacy
#     # NOTE: u_unnorm should be in order x, y already, due to pytorch convention.
#     # TODO: Normalize u
#     outdir = '/tmp/pose'
#     if first and os.path.exists(outdir):
#         print("Removing dir")
#         for fname in os.listdir(outdir):
#             os.remove(os.path.join(outdir, fname))
#         first = False
#     else:
#         print("Not removing dir")
#     os.makedirs(outdir, exist_ok=True)
# 
#     # plt.figure()
#     # plt.imshow(img)
#     # plt.savefig(os.path.join(outdir, '{}{}_img.jpg'.format(prefix, obj)))
# 
#     plt.figure()
#     plt.scatter(u_unnorm[1,:], 480-u_unnorm[0,:])
#     plt.xlim((0, 640))
#     plt.ylim((0, 480))
#     plt.savefig(os.path.join(outdir, '{}{}_2d.png'.format(prefix, obj)))
# 
#     plt.figure()
#     plt.xlim((-0.13, 0.13))
#     plt.ylim((-0.13, 0.13))
#     plt.scatter(U[0,:], U[1,:])
#     plt.savefig(os.path.join(outdir, '{}{}_z_3d.png'.format(prefix, obj)))
# 
#     plt.figure()
#     plt.xlim((-0.13, 0.13))
#     plt.ylim((-0.13, 0.13))
#     plt.scatter(U[1,:], U[2,:])
#     plt.savefig(os.path.join(outdir, '{}{}_x_3d.png'.format(prefix, obj)))
# 
#     plt.figure()
#     plt.xlim((-0.13, 0.13))
#     plt.ylim((-0.13, 0.13))
#     plt.scatter(U[0,:], U[2,:])
#     plt.savefig(os.path.join(outdir, '{}{}_y_3d.png'.format(prefix, obj)))
# 
#     plt.figure()
#     plt.xlim((-0.13, 0.13))
#     plt.ylim((-0.13, 0.13))
#     plt.scatter(mesh[0,:], mesh[1,:])
#     plt.savefig(os.path.join(outdir, '{}{}_z_mesh.png'.format(prefix, obj)))
# 
#     plt.figure()
#     plt.xlim((-0.13, 0.13))
#     plt.ylim((-0.13, 0.13))
#     plt.scatter(mesh[1,:], mesh[2,:])
#     plt.savefig(os.path.join(outdir, '{}{}_x_mesh.png'.format(prefix, obj)))
# 
#     plt.figure()
#     plt.xlim((-0.13, 0.13))
#     plt.ylim((-0.13, 0.13))
#     plt.scatter(mesh[0,:], mesh[2,:])
#     plt.savefig(os.path.join(outdir, '{}{}_y_mesh.png'.format(prefix, obj)))
# 
#     # Axes3D.scatter
#     x_pred = None
#     ddU = None
#     return x_pred, ddU
