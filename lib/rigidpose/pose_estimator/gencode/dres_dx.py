import numpy as np
import math

def calc_dres_dx(x, U):
    """
    Inputs:
        Pose parameters x = [alpha, beta, delta, tx, ty, tz] --- (6 x 1) array
        3D-points U --- (3 x N) array
    Outputs:
        dres_dx --- (2N x 6) array
    """
    assert x.shape == (1,6)
    assert U.shape[0] == 4
    N = U.shape[1]
    alpha, beta, delta, tx, ty, tz = np.squeeze(x)
    Ux, Uy, Uz = U[0,:,np.newaxis], U[1,:,np.newaxis], U[2,:,np.newaxis]

    t1 = math.cos(beta)
    t2 = Uy - ty
    t3 = t1 * t2
    t4 = Ux - tx
    t5 = math.cos(delta)
    t6 = t5 ** 2
    t7 = t4 * t6
    t9 = 2 * t3 * t7
    t10 = math.sin(delta)
    t11 = ty + Ux - tx - Uy
    t13 = -ty + Ux - tx + Uy
    t16 = math.sin(beta)
    t17 = Uz - tz
    t18 = t16 * t17
    t20 = t10 * t11 * t13 * t1 + t18 * t2
    t22 = t16 * t10
    t24 = -t22 * t17 + t3
    t25 = t4 * t24
    t27 = math.cos(alpha)
    t30 = t11 * t13
    t31 = t30 * t6
    t33 = t1 ** 2
    t42 = t10 * t17
    t46 = tz + Uy - ty - Uz
    t48 = -tz + Uy - ty + Uz
    t53 = math.sin(alpha)
    t56 = t33 - 2
    t58 = t13 * t6
    t59 = t56 * t11 * t58
    t60 = t10 * t2
    t62 = t18 * t1
    t69 = t46 * t48 * t33
    t70 = t17 * t2
    t71 = t70 * t1
    t73 = 2 * t22 * t71
    t75 = t27 ** 2
    t77 = t16 * t2
    t84 = t42 * t1 + t77
    t87 = (2 * t77 * t7 + (t22 * t30 - t71) * t5 - t4 * t84) * t53
    t89 = 2 * t87 * t27
    t90 = t4 * t5
    t92 = 2 * t60 * t90
    t93 = t4 ** 2
    t95 = 0.1e1 / ((t59 - 2 * t4 * (t60 * t33 - 2 * t60 - t62) * t5 + t69 - t73 + t30) * t75 + t89 + t31 - t92 - t93)
    drd1 = ((t20 * t5 - t25 + t9) * t27 + (t1 * t16 * t31 - 2 * t4 * (t17 * t33 + t22 * t3 - Uz / 2 + tz / 2) * t5 + 2 * t42 * t2 * t33 + t16 * t46 * t48 * t1 - t42 * t2) * t53) * t95
    t96 = Uy ** 2
    t98 = 2 * Uy * ty
    t99 = ty ** 2
    t100 = t17 ** 2
    drd2 = ((-t31 + t92 - t96 + t98 - t99 - t100) * t27 + t87) / ((t59 - 2 * t4 * (t56 * t2 * t10 - t62) * t5 - t73 + t69 + t30) * t75 + t89 + t31 - t92 - t93)
    t116 = t10 * t4 + t2 * t5
    t120 = Ux ** 2
    t123 = tx ** 2
    t124 = -2 * Ux * tx + t120 + t123 + t96 - t98 + t99
    drd3 = (t116 * t17 * t27 + (t124 * t1 + t18 * t90 - t22 * t70) * t53) * t95
    drd4 = (t5 * t17 * t27 + t24 * t53) * t95
    t136 = t1 * t4
    drd5 = (-t42 * t27 - (t18 * t5 + t136) * t53) * t95
    drd6 = ((-t4 * t5 + t60) * t27 + t116 * t16 * t53) * t95
    dsd1 = (-t33 * t11 * t58 + 2 * t136 * (t1 * t10 * t2 - t18) * t5 - t69 + t73 + t96 - t98 + t99 + t93) * t95
    dsd2 = (-t20 * t5 + t25 - t9) * t95
    t159 = t1 * t17
    dsd3 = (t16 * t124 - t159 * t90 + t42 * t3) * t95
    dsd4 = t84 * t95
    dsd5 = (t159 * t5 - t16 * t4) * t95
    dsd6 = -t1 * t116 * t95

    dres_dx = np.zeros((2*N, 6))
    dres_dx[::2, :] = np.concatenate([drd1, drd2, drd3, drd4, drd5, drd6], axis=1)
    dres_dx[1::2, :] = np.concatenate([dsd1, dsd2, dsd3, dsd4, dsd5, dsd6], axis=1)
    return dres_dx
