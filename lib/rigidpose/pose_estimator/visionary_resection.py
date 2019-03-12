import numpy as np
from scipy import linalg

def resec3pts(u, U, coord_change=True):
    # Assert homogeneous coordinates
    assert u.shape[0] == 3
    assert U.shape[0] == 4

    if coord_change:
        t = np.mean(U[0:3,:], axis=1)
        scale = np.max(np.abs(U))
        T = np.diag([1./scale]*3+[1.])
        T[0:3,3] = -t/scale
        Up = np.dot(T, U)
        # Recursive function call:
        pdata = resec3pts(u, Up, coord_change=False)
        return [np.dot(PP,T)*scale for PP in pdata]

    pdata = []

    # Compute gramians
    Udiff = U[0:3, 1:3] - U[0:3, (0,0)]
    G = np.dot(Udiff.T, Udiff)
    G11 = G[0,0]
    G12 = G[0,1]
    G22 = G[1,1]
    ug = np.dot(u.T, u)
    u11 = ug[0,0]
    u22 = ug[1,1]
    u33 = ug[2,2]
    u23 = ug[1,2]
    u12 = ug[0,1]
    u13 = ug[0,2]

    # Regard the depths l1, l2 and l3 as unknowns.
    # The following equations should be fulfilled:
    # t0 = l2^2*u22+l1^2*u11-2*l1*l2*u12-G11
    # t0 = l3^2*u33+l1^2*u11-2*l1*l3*u13-G22
    # t0 = l3*l2*u23+l1^2*u11-l1*l2*u12-l1*l3*u13-G12

    # Eliminate l2 and l3 to get an 8th-degree polynomial
    # with even terms only: l1^8, l1^6, l1^4, l1^2, 1.
    l1poly = [0]*5 # List will be filled with coefficients
    t1 = u11**2
    t2 = t1**2
    t3 = u23**2
    t4 = t3**2
    t6 = u22**2
    t8 = u13**2
    t9 = t8**2
    t11 = t1*u11
    t13 = t8*t3
    t15 = t1*u22
    t21 = t8*u33
    t23 = u12**2
    t48 = u33**2
    t55 = t23**2
    l1poly[0] = t2*t4+t1*t6*t9+2*t11*u22*t13-4*t15*t8*u13*u12*u23-2*t11*t6*t21+2*t15*t21*t23-4*t11*t3*u23*u12*u13-2*t2*t3*u33*u22+2*t11*t3*u33*t23+4*t1*t23*t13+4*t11*u12*u13*u23*u33*u22-4*t1*t23*u12*u13*u23*u33+t48*t6*t2-2*t48*u22*t11*t23+t48*t1*t55
    t1 = u11**2
    t2 = t1*u12
    t3 = u23**2
    t4 = t3*u23
    t10 = u13**2
    t11 = t10*u13
    t23 = u12**2
    t30 = t10**2
    t34 = t1*u11
    t36 = t3*u33
    t37 = t36*u22
    t39 = t10*G11
    t40 = t23**2
    t43 = u22**2
    t45 = t10*u33
    t48 = t3**2
    t49 = t34*t48
    t54 = G12*u22
    t58 = t23*u11
    t61 = u33**2
    t62 = t61*u22
    t63 = t1*G11
    t66 = t23*u12
    t69 = u23*u33
    t73 = u11*u22*t10
    t75 = u33*t23*G12
    t82 = G22*u12*u13
    t88 = t1*u22
    t91 = -2*t49*G11+8*t30*t23*t54+2*t34*G11*t37-12*t39*t58*t3-2*t62*t63*t23+4*u11*t66*u13*t69*G12-16*t73*t75+6*t73*u33*G11*t23+8*t1*t4*t82+4*t34*t3*u33*u22*G12-2*t88*t39*t3
    t102 = t61*u11
    t112 = G22*t66
    t117 = u11*G12
    t124 = 2*u11*t43*t30*G22-4*t61*t43*t34*G12+8*t62*t1*t23*G12-4*t102*t40*G12+2*t102*t40*G11+8*t11*G11*t66*u23-4*t82*t69*t88+4*t112*u13*t69*u11-4*t30*t43*t117-4*G22*t40*t45-4*G22*t23*t30*u22
    t158 = 8*t112*t11*u23+8*t10*t40*G12*u33+16*t10*t23*t117*t3-16*t11*t66*G12*u23-4*t63*u23*u12*u33*u22*u13-2*G22*t1*t36*t23+8*t2*u13*t4*G11-12*t58*t10*t3*G22-4*t2*u13*t69*t54-2*G22*t43*t45*t1-4*t1*t3*t75
    l1poly[1]  = -4*t2*G12*t4*u13+4*u11*G11*u23*u12*t11*u22+4*t11*u22*u11*G12*u12*u23+6*G22*u22*t10*u33*u11*t23-4*t10*u22*t1*G12*t3-4*t30*G11*t23*u22+2*G22*t34*t37-4*t39*t40*u33+8*t1*t43*t45*G12-2*t49*G22+t91+t124+t158
    t1 = G22*G11
    t3 = u13**2
    t4 = t3*u13
    t5 = u12*t4
    t10 = u23**2
    t11 = t10*u23
    t12 = t11*u13
    t15 = u11*u12
    t19 = u33**2
    t20 = u22**2
    t22 = u11**2
    t23 = G12**2
    t28 = u12**2
    t30 = u33*t28*G12
    t32 = t3**2
    t35 = G22**2
    t39 = G22*t20
    t42 = G22*u22
    t47 = t35*u22
    t53 = t15*u13
    t54 = u23*u33
    t58 = -4*t1*u23*t5*u22+4*G22*u12*G12*t12*u11+4*t15*G12*t12*G11+6*t19*t20*t22*t23+4*G22*u11*t10*t30+4*t32*t20*t23+4*t35*t28*t10*t3-4*t39*t32*G12+12*t42*t4*G12*u12*u23-4*t47*t5*u23+2*t47*t3*u11*t10+4*t53*t54*u22*t23
    t62 = t3*u22
    t63 = t23*u33
    t68 = u33*G11
    t73 = G12*G11
    t74 = t73*t10
    t80 = t28*u12
    t85 = t10**2
    t87 = G11**2
    t91 = t28**2
    t93 = t19*t91
    t97 = G22*t28
    t98 = t3*G11
    t105 = -12*t53*t11*G22*G11+12*t62*t63*t28+8*t42*u11*u13*t68*u23*u12-8*t3*t28*t74+4*u11*t28*t10*t23*u33-8*u13*t80*t63*u23+t35*t22*t85+t22*t87*t85+t19*t87*t91+4*t93*t23+t35*t20*t32+12*t97*t98*t10-4*u13*t87*t80*u33*u23
    t108 = u33*u22
    t109 = t108*G12
    t113 = t11*u12*u13
    t115 = u11*t87
    t130 = t42*t3
    t143 = -4*t98*t28*t109-4*t35*u11*t113-4*t115*t113-2*t1*t10*t108*t22-4*G22*t80*u13*t54*G11-4*t22*G11*t10*t109-8*t97*G12*t10*t3-2*t130*t68*t28-4*t130*G12*u11*t10+2*t115*t10*u33*t28-2*t22*t10*t108*t23-4*t93*t73
    t146 = t19*u22
    t153 = t23*u11
    t184 = 4*t62*u11*t74+4*t146*u11*t73*t28+4*t3*t87*t28*t10+4*t62*t153*t10+4*t22*t85*t1-10*u11*t20*t3*u33*t23-8*t4*u22*t23*u12*u23-4*t130*t30-4*u11*G11*t10*t30-10*t146*t153*t28+12*u13*G11*t80*u33*G12*u23-4*G22*t22*t10*t109+4*t39*t3*u33*u11*G12
    l1poly[2]  = t58+t105+t143+t184
    t1 = G22*u22
    t6 = u33*G11*u23*u12
    t8 = u12**2
    t10 = G12**2
    t11 = u23**2
    t15 = G22*u12
    t17 = t11*u23
    t19 = t17*u13*G11
    t21 = u13**2
    t23 = t10*G11
    t26 = u33**2
    t27 = t26*u22
    t28 = t10*G12
    t31 = u13*u22
    t38 = G11**2
    t39 = G22*t38
    t45 = u33*u22
    t46 = t45*t10
    t53 = -8*t1*u13*G12*t6-4*G22*t8*t10*t11*u33-4*t15*G12*t19-4*t21*u22*t23*t11+4*t27*t28*t8-4*t31*t28*u33*u12*u23+4*t31*t10*t6+4*t39*t17*u12*u13+2*u11*G11*t11*t46+2*G22*u11*t11*t46-2*t27*t23*t8
    t54 = G22**2
    t59 = u22**2
    t75 = G22*G11*t11
    t93 = t11**2
    t99 = -2*t54*u22*t21*G11*t11+4*t21*t59*t28*u33+4*t54*u12*t19+4*t1*t21*G12*G11*t11+4*t15*u13*u23*u33*u22*t10+4*t75*u33*t8*G12-2*t39*t11*u33*t8+4*t75*t45*u11*G12-2*G22*t59*t21*u33*t10-4*t26*t59*u11*t28-2*u11*t38*t93*G22-2*t54*u11*t93*G11
    l1poly[3]  = t53+t99
    t1 = G22**2
    t2 = G11**2
    t4 = u23**2
    t5 = t4**2
    t7 = u33**2
    t8 = u22**2
    t10 = G12**2
    t11 = t10**2
    l1poly[4] = t1*t2*t5+t7*t8*t11-2*G22*G11*t4*u33*u22*t10

    # Solve 4th degree polynomial in (l1^2)
    l1list = np.roots(l1poly)
    l1list = l1list[np.isreal(l1list)]
    assert np.all(np.isreal(l1list))
    l1list = np.real(l1list)
    l1list = l1list[l1list > 0]
    l1list = np.sqrt(l1list)

    l1list = list(l1list)
    l2list = []
    l3list = []
    pdata = []
    for l1 in l1list:
        # Solve for l2
        t1 = l1**2
        t2 = t1**2
        t3 = t2*u11
        t4 = u13**2
        t6 = t4*u13*u22
        t8 = u11**2
        t9 = t2*t8
        t10 = u23**2
        t14 = t4*u12*u23
        t16 = u33*u22
        t19 = u12**2
        t23 = t1*u11
        t25 = u13*G11*t10
        t27 = t23*u33
        t29 = u22*u13*G12
        t35 = G11*u23*u12
        t37 = t1*G22
        t41 = u13*t1
        t46 = u33*G11*t19
        t49 = G11*t10
        t51 = u13*u33
        t52 = G12**2
        t53 = u22*t52
        t58 = G12*t1
        t68 = -2*t37*t14+t41*G22*u11*t10-t41*t46-u13*G22*t49-t51*t53+2*u33*u12*t52*u23-2*t58*t6-2*t58*u13*u11*t10+4*t58*t14+2*G12*u13*t49-2*G12*u33*t35
        t71 = u23*t2
        t75 = t10*u23
        t78 = u12*u13*t10
        t87 = u23*t1
        t95 = u22*u11
        t103 = t2*u12
        t114 = t1*u12
        t118 = t71*u22*t4*u11+t9*t75-4*t3*t78-t71*t16*t8+t71*u33*u11*t19-t23*G11*t75-t87*G22*u22*t4-t37*u11*t75+2*t37*t78+2*t87*u33*t95*G12-t87*t46+G22*G11*t75-u23*u33*t53-2*t103*t6+4*t2*t19*t4*u23+2*t103*u33*t95*u13-2*t2*t19*u12*t51+2*t114*t25-2*t114*u33*t29
        l2 = -l1*(t3*t6+t9*u13*t10-2*t3*t14-t9*t16*u13+t3*u33*t19*u13-t23*t25+2*t27*t29-2*t27*u12*G12*u23+2*t27*t35+t37*t6+t68)/t118

        # Solve for l3
        t1 = l1**2
        t2 = t1**2
        t3 = t2*u11
        t4 = u13**2
        t7 = u11**2
        t8 = t2*t7
        t9 = u23**2
        t14 = t1*u11
        t18 = t1*t4
        t21 = u11*t9
        t24 = t1*u13
        t29 = u33*u22
        t33 = G12**2
        t36 = u12**2
        t42 = u33*G11
        t45 = t3*u22*t4+t8*t9-2*t3*u12*u13*u23-t14*G11*t9-G22*u22*t18-G22*t1*t21+2*G22*u12*t24*u23+G22*G11*t9-t29*t8+2*t29*t14*G12-t29*t33+u33*t2*u11*t36-2*u33*t36*t1*G12+t42*t36*t1
        t52 = u12*u23
        t54 = t1*u33
        l3 = -t45/l1/(-t1*t4*u13*u22-t24*t21+2*t18*t52+t54*u22*u11*u13-t54*t36*u13+u13*G11*t9-t29*u13*G12+u33*u12*G12*u23-t42*t52)/2

        l2list.append(l2)
        l3list.append(l3)

        # Compute R & t such that
        # R*U+t=u*diag([l1 l2 l3])
        ut = np.dot(u, np.diag([l1, l2, l3]))
        Ut = U[0:3,:]
        t1 = Ut[0:3,0,np.newaxis]
        t2 = ut[0:3,0,np.newaxis]
        Ut = Ut[0:3,:] - np.dot(t1, np.ones((1,3)))
        ut = ut - np.dot(t2, np.ones((1,3)))
        R1, _, _ = linalg.svd(Ut[0:3,1,np.newaxis])
        if linalg.det(R1) < 0:
            R1 = np.dot(R1, np.diag([1, 1, -1]))
        R2, _, _ = linalg.svd(ut[0:3,1,np.newaxis])
        if linalg.det(R2) < 0:
            R2 = np.dot(R2, np.diag([1, 1, -1]))
        if linalg.norm(np.dot(R1.T, Ut[0:3,1,np.newaxis]) - np.dot(R2.T, ut[0:3,1,np.newaxis]), ord=2) > 0.001:
            R2 = np.dot(R2, np.diag([-1, -1, 1]))
        Ut = np.dot(R1.T, Ut)
        ut = np.dot(R2.T, ut)
        R11, _, _ = linalg.svd(Ut[1:3,2,np.newaxis])
        if linalg.det(R11) < 0:
            R11 = np.dot(R11, np.diag([1, -1])) #corrected by Kahl
        R22, _, _ = linalg.svd(ut[1:3,2,np.newaxis])
        if linalg.det(R22) < 0:
            R22 = np.dot(R22, np.diag([1, -1])) #corrected by Kahl
        if linalg.norm(np.dot(R11.T, Ut[1:3,2]) - np.dot(R22.T, ut[1:3,2]), ord=2) > 0.001:
            R22 = np.dot(R22, np.diag([-1, -1]))
        R11 = linalg.block_diag(1, R11)
        R22 = linalg.block_diag(1, R22)
        Ut = np.dot(R11.T, Ut)
        ut = np.dot(R22.T, ut)
        T1 = np.eye(4); T1[0:3,3] = -t1.squeeze()
        RR = np.dot(np.dot(np.dot(R2, R22), R11.T), R1.T)
        RR = linalg.block_diag(RR, 1)
        T2 = np.eye(4); T2[0:3,3] = t2.squeeze()
        T = np.dot(np.dot(T2, RR), T1)
        PP = T[0:3,:]

        pdata.append(PP)

    # llists = np.array([l1list, l2list, l3list]).T

    return pdata
