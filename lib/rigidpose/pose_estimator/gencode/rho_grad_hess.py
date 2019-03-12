import numpy as np

def calc_rho_grad_hess(res, alpha_rho, c, w=None):
    """
    Residuals (1d-array): res
    Jon Barron parameters (scalar): alpha_rho, c
    Outputs: 1d-array derivatives
    """
    alpha = alpha_rho
    x = res
    assert len(res.shape) == 1
    if w is None:
        w = np.ones(res.size//2)
    assert len(w.shape) == 1
    assert res.size == w.size*2
    w = np.kron(w, [1, 1])

    t1 = -2 + alpha
    t2 = x ** 2
    t3 = c ** 2
    t5 = t3 * t1 - t2
    t9 = t5 / t3 / t1
    t11 = t9 ** (alpha / 2)
    r = -t1 * (t11 - 1) / alpha
    t16 = t11 * x
    t18 = t1 / t5
    drdx = t16 * t18
    t19 = t3 * alpha
    t20 = 2 * t3
    t21 = t19 - t20 - t2
    t23 = np.log(t9)
    t25 = t2 * alpha
    t26 = 4 * t3
    t35 = alpha ** 2
    drda = (-(alpha * t21 * t23 + 2 * t2 + t25 + t26) * t1 * t11 + 4 * t19 - 8 * t3 - 4 * t2) / t35 / t21 / 2
    t41 = 0.1e1 / c
    drdc = -t11 * t2 * t18 * t41
    t48 = t21 ** 2
    dr2dx2 = ((t3 - t2) * alpha - t20 + t2) * t11 * t1 / t48
    t54 = t5 ** 2
    t55 = 1 / t54
    dr2dxda = t16 * (t5 * t23 + t2) * t1 * t55 / 2
    dr2dxdc = -(2 * t19 - t25 - t26) * t11 * x * t1 * t55 * t41

    return {
        'rho': r*w,
        'drho_dres': drdx*w,
        'drho_da': drda*w,
        'drho_dc': drdc*w,
        'drho2_dres2': dr2dx2*w,
        'drho2_dresda': dr2dxda*w,
        'drho2_dresdc': dr2dxdc*w,
    }
