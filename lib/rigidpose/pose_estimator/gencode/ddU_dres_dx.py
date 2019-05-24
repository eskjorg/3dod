import numpy as np
import math

def calc_ddU_dres_dx(x, U):
    """
    Inputs:
        Pose parameters x = [alpha, beta, delta, tx, ty, tz] --- (6 x 1) array
        3D-points U --- (3 x N) array
    Outputs (dres_dU, ddU_dres_dx), where:
        dres_dU --- (2N x 3) array
        ddU_dres_dx --- (2N x 6 x 3) array
    """
    assert x.shape == (1,6)
    assert U.shape[0] == 4
    N = U.shape[1]
    alpha, beta, delta, tx, ty, tz = np.squeeze(x)
    Ux, Uy, Uz = U[0,:], U[1,:], U[2,:]

    t1 = math.cos(delta)
    t2 = Uz - tz
    t3 = t1 * t2
    t4 = math.cos(alpha)
    t6 = math.sin(alpha)
    t7 = math.cos(beta)
    t8 = Uy - ty
    t9 = t7 * t8
    t10 = math.sin(beta)
    t11 = math.sin(delta)
    t12 = t10 * t11
    t13 = t12 * t2
    t14 = t9 - t13
    t17 = t7 ** 2
    t18 = t17 - 2
    t19 = -ty + Ux - tx + Uy
    t21 = ty + Ux - tx - Uy
    t22 = t1 ** 2
    t23 = t21 * t22
    t24 = t18 * t19 * t23
    t25 = Ux - tx
    t26 = t11 * t8
    t27 = t26 * t17
    t28 = t10 * t2
    t29 = t28 * t7
    t30 = 2 * t26
    t35 = -tz + Uy - ty + Uz
    t36 = tz + Uy - ty - Uz
    t37 = t35 * t36
    t39 = t2 * t8
    t40 = t39 * t7
    t41 = t12 * t40
    t43 = t19 * t21
    t45 = t4 ** 2
    t47 = t10 * t8
    t48 = t25 * t22
    t49 = t47 * t48
    t51 = t12 * t43
    t54 = t11 * t2
    t55 = t54 * t7
    t56 = t55 + t47
    t57 = t56 * t25
    t59 = t6 * (2 * t49 + (-t40 + t51) * t1 - t57)
    t62 = t43 * t22
    t63 = t25 * t1
    t65 = 2 * t26 * t63
    t66 = t25 ** 2
    t68 = 0.1e1 / ((t24 - 2 * t25 * (t27 - t29 - t30) * t1 + t37 * t17 - 2 * t41 + t43) * t45 + 2 * t59 * t4 + t62 - t65 - t66)
    drdUx = (-t6 * t14 - t3 * t4) * t68
    t72 = t28 * t1 + t7 * t25
    drdUy = (t54 * t4 + t72 * t6) * t68
    t78 = t11 * t25
    t79 = t8 * t1 + t78
    t80 = t10 * t79
    t81 = t80 * t6
    drdUz = ((t63 - t26) * t4 - t81) * t68
    dsdUx = -t56 * t68
    t84 = t7 * t2
    t86 = t10 * t25
    t87 = -t84 * t1 + t86
    dsdUy = t87 * t68
    dsdUz = t79 * t7 * t68
    t89 = t18 * t2
    t90 = t8 * t22
    t91 = t89 * t90
    t92 = t54 * t17
    t93 = t47 * t7
    t94 = 2 * t54
    t95 = t92 + t93 - t94
    t96 = t25 * t95
    t98 = t39 * t17
    t99 = 2 * t98
    t100 = t37 * t7
    t101 = t12 * t100
    t104 = t28 * t48
    t105 = 2 * t104
    t106 = t12 * t39
    t107 = 2 * t106
    t110 = t7 * t11
    t111 = t110 * t8
    t112 = t111 - t28
    t113 = t25 * t112
    t117 = t17 - 0.1e1 / 0.2e1
    t119 = t2 * t22
    t129 = 2 * t39
    t131 = t7 + 2
    t133 = t10 * t131 * t25
    t134 = Uy ** 2
    t135 = 3 * t134
    t136 = Uy * ty
    t137 = 6 * t136
    t138 = ty ** 2
    t139 = 3 * t138
    t140 = -t135 + t137 - t139 + t66
    t141 = t7 - 2
    t143 = t22 * t1
    t146 = t17 * t7
    t149 = 3 * t43 * t2 * t146
    t150 = t134 / 3
    t151 = 0.2e1 / 0.3e1 * t136
    t152 = t138 / 3
    t153 = -t150 + t151 - t152 + t66
    t154 = t10 * t153
    t157 = t43 * t84
    t158 = 6 * t157
    t159 = t154 * t26
    t165 = 2 * t54 * t8 * t146
    t167 = t36 * t17
    t168 = t10 * t35 * t167
    t169 = t54 * t9
    t170 = 4 * t169
    t176 = 2 * t136
    t177 = t2 ** 2
    t182 = 3 * (t134 - t176 + t138 - t177 / 3) * t2 * t146
    t186 = t10 * (t134 - t176 + t138 - 3 * t177) * t27
    t187 = 3 * t157
    t190 = t45 * t4
    t192 = t17 - 0.4e1 / 0.3e1
    t194 = t8 * t143
    t197 = t140 * t11
    t199 = t28 * t9
    t200 = 4 * t199
    t205 = t134 / 2
    t206 = t138 / 2
    t207 = Ux ** 2
    t208 = Ux * tx
    t209 = 2 * t208
    t210 = tx ** 2
    t211 = t177 / 2
    t220 = 2 * t12 * t19 * t21 * t2 * t7
    t221 = t153 * t8
    t226 = t11 * t35 * t167
    t227 = 3 * t226
    t228 = 6 * t199
    t242 = 2 * t169
    t243 = 2 * t134
    t244 = 4 * t136
    t245 = 2 * t138
    t252 = t47 * t11
    t253 = t84 + t252
    t255 = 3 * t66 * t253
    t261 = t11 * t22
    t265 = 3 * t66 * t8 * t1
    t267 = t11 * t66 * t25
    t271 = 0.1e1 / ((t133 * t140 * t141 * t143 + (-3 * t154 * t27 - t149 + t158 + 12 * t159) * t22 + 3 * (t10 * t140 + t165 + t168 - t170) * t25 * t1 - t182 - t186 - t187 - 3 * t159) * t190 - 3 * (3 * t153 * t192 * t194 + (t197 * t17 + t200 - 0.4e1 / 0.3e1 * t197) * t25 * t22 + (-2 * (-t205 + t136 - t206 + t207 - t209 + t210 + t211) * t8 * t17 + t220 + 3 * t221) * t1 + t25 * (t227 - t228 + t197) / 3) * t6 * t45 + (3 * t86 * t140 * t143 + (-t187 - 9 * t159) * t22 - 3 * (-t242 + t10 * (-t243 + t244 - t245 + t66)) * t25 * t1 + t255) * t4 - t6 * (t25 * t140 * t261 + 3 * t221 * t143 - t265 - t267))
    drd11 = ((t96 * t1 - t101 + t39 + t91 - t99) * t45 - t6 * (t105 + (t100 - t107) * t1 + t113) * t4 - 2 * t117 * t8 * t119 - 2 * (t92 + t93 - t54 / 2) * t25 * t1 + 4 * t98 + 2 * t101 - t129) * t271
    t272 = t89 * t48
    t273 = t8 * t17
    t274 = t54 * t273
    t275 = -tz + Ux - tx + Uz
    t277 = tz + Ux - tx - Uz
    t278 = t277 * t7
    t279 = t10 * t275 * t278
    t280 = t54 * t8
    t281 = 2 * t280
    t284 = t2 * t17
    t285 = t12 * t9
    t290 = t28 * t90
    t291 = 2 * t290
    t292 = 2 * t13
    t298 = t28 * t8
    t302 = t25 * t117
    t313 = 2 * t25 * (t284 + t285 - Uz / 2 + tz / 2)
    t315 = t8 ** 2
    t317 = t207 - t209 + t210 - 3 * t315
    t322 = t207 - t209 + t210 - t315 / 3
    t323 = t10 * t322
    t326 = t323 * t26
    t341 = t317 * t11
    t352 = t322 * t8
    t369 = 2 * t315
    t385 = 0.1e1 / ((t133 * t317 * t141 * t143 + (-3 * t323 * t27 - t149 + t158 + 12 * t326) * t22 + 3 * (t10 * t317 + t165 + t168 - t170) * t25 * t1 - t182 - t186 - t187 - 3 * t326) * t190 - 3 * t6 * (3 * t322 * t192 * t194 + (t341 * t17 + t200 - 0.4e1 / 0.3e1 * t341) * t25 * t22 + (-2 * (t207 - t209 + t210 - t37 / 2) * t8 * t17 + t220 + 3 * t352) * t1 + t25 * (t227 - t228 + t341) / 3) * t45 + (3 * t86 * t317 * t143 + (-t187 - 9 * t326) * t22 - 3 * (-t242 + t10 * (t207 - t209 + t210 - t369)) * t25 * t1 + t255) * t4 - t6 * (t25 * t317 * t261 + 3 * t352 * t143 - t265 - t267))
    drd12 = ((t272 + (-t274 - t279 + t281) * t1 + t25 * (t284 + t285 + Uz - tz)) * t45 + (t291 + t25 * (t9 + t292) * t1 + t11 * t275 * t278 - t298) * t6 * t4 - 2 * t302 * t119 + (2 * t274 + 2 * t279 - t280) * t1 - t313) * t385
    t386 = t18 * t8
    t389 = t11 * t19
    t391 = t389 * t21 * t17
    t392 = t389 * t21
    t396 = t12 * t84
    t403 = t10 * t19
    t404 = t403 * t23
    t410 = t169 / 2
    t420 = 2 * t199
    t431 = t207 - t209 - t135 + t137 + t210 - t139
    t435 = t207 - t209 - t150 + t151 + t210 - t152
    t436 = t10 * t435
    t439 = t436 * t26
    t454 = t11 * t431
    t468 = t435 * t8
    t500 = 0.1e1 / ((t10 * t141 * t131 * t25 * t431 * t143 + (-3 * t436 * t27 - t149 + t158 + 12 * t439) * t22 + 3 * (t10 * t431 + t165 + t168 - t170) * t25 * t1 - t182 - t186 - t187 - 3 * t439) * t190 - 3 * t6 * (3 * t192 * t435 * t194 + (t454 * t17 + t200 - 0.4e1 / 0.3e1 * t454) * t25 * t22 + (-2 * (t207 - t209 - t205 + t136 + t210 + (-ty + Uz - tz) * (ty + Uz - tz) / 2) * t8 * t17 + t220 + 3 * t468) * t1 + t25 * (t227 - t228 + t454) / 3) * t45 + (3 * t86 * t431 * t143 + (-t187 - 9 * t439) * t22 - 3 * (-t242 + t10 * (t207 - t209 - t243 + t244 + t210 - t245)) * t25 * t1 + t255) * t4 - t6 * (t78 * t431 * t22 + 3 * t468 * t143 - t265 - t267))
    drd13 = ((-2 * t386 * t48 + (-t391 - t199 + 2 * t392) * t1 + (t273 - t396 - 2 * Uy + 2 * ty) * t25) * t45 + 2 * t6 * (t404 - (t84 + 4 * t252) * t25 * t1 / 2 + t410 - t403 * t21 / 2) * t4 + 4 * t302 * t90 + (2 * t391 + t420 - t392) * t1 - 2 * t25 * (t273 - t396 - Uy / 2 + ty / 2)) * t500
    t501 = t84 * t48
    t502 = 2 * t177
    t503 = t134 - t176 + t138 + t502
    t507 = t25 * t253
    t510 = t7 * t10
    t512 = t510 * t39 * t22
    t514 = t25 * (t273 - t396 + Uy - ty)
    t521 = t56 * t8 * t1
    drd21 = ((t501 + (t10 * t503 - t169) * t1 + t507) * t45 + t6 * (t514 * t1 - t503 * t11 - t226 + t420 - t512) * t4 + t501 - t521 - t507) * t271
    t523 = t84 * t90
    t533 = t510 * t2 * t25 * t22
    t535 = t275 * t277 * t17
    t543 = t79 * t87
    drd22 = ((-t523 - t57 * t1 + 2 * t40 - t10 * (t207 - t209 + t210 + t502) * t11) * t45 - (t533 + (t535 - t41 + t207 - t209 + t210 + t502) * t1 - t25 * (t27 - t29 + t26)) * t6 * t4 + t543) * t385
    t553 = t8 * t25
    t554 = t553 * t22
    t564 = t62 - t65 - t66
    drd23 = ((-t7 * t19 * t23 + 2 * t113 * t1 + (-t207 + t209 - t210 - t369) * t7 + t107) * t45 + t6 * (2 * t510 * t554 + (t12 * t43 * t7 + t129 - t98) * t1 - t96) * t4 - t7 * t564) * t385
    t570 = t134 - t176 + t138 + t211
    drd31 = ((2 * t25 * t14 * t1 - 2 * t570 * t11 * t7 - t291 + t298) * t45 + t6 * (t272 + (-2 * t10 * t570 * t7 - t274 + t281) * t1 - t313) * t4 + t290 - 2 * (t9 - t13 / 2) * t25 * t1 + 2 * t8 * t112) * t271
    t592 = 2 * t207
    t593 = 4 * t208
    t594 = 2 * t210
    t599 = t28 / 2
    t612 = t207 - t209 + t210 + t211
    drd32 = ((-t105 + ((-t592 + t593 - t594 - t177) * t7 + t107) * t1 + 2 * (t111 + t599) * t25) * t45 + 2 * t6 * (-t91 / 2 - t25 * (t92 - 2 * t93 - t94) * t1 / 2 - t98 / 2 + t10 * t612 * t110 - t39 / 2) * t4 + t104 + (2 * t66 * t7 - t106) * t1 - 2 * t25 * (t111 - t599)) * t385
    t632 = 2 * t47
    t650 = t389 * t21 * t1 - t553 + 2 * t554
    drd33 = ((4 * t49 + (t40 + 2 * t51) * t1 + (t55 - t632) * t25) * t45 - t6 * (t24 - 2 * (t27 + t29 / 2 - t30) * t25 * t1 + (-t592 + t593 - t134 + t176 - t594 - t138) * t17 + t41 + t43) * t4 - t10 * t650) * t500
    t653 = t28 * t22
    t654 = 4 * t653
    t655 = 2 * t111
    t656 = 2 * t28
    drd41 = ((-t654 - t655 + t656) * t45 - 2 * t6 * t95 * t1 * t4 + 2 * t653 + t655 - t656) * t271
    t670 = t110 * t25
    t676 = 2 * Uz
    t677 = 2 * tz
    drd42 = (((-t7 * t8 + 4 * t13) * t1 + t670) * t45 + t6 * (-2 * t89 * t22 + t510 * t63 + t284 + t285 - t676 + t677) * t4 + (t9 - t292) * t1 - t670) * t500
    t684 = t86 * t22
    t686 = 2 * t252
    t687 = t84 - t686
    t698 = -t26 * t1 - Ux + t48 + tx
    drd43 = ((t687 * t1 + 2 * t684 - t86) * t45 + (t11 * t18 * t63 + t386 * t22 + Uy + t273 - t396 - ty) * t6 * t4 - t10 * t698) * t500
    drd51 = drd42
    drd52 = ((2 * t7 * t1 * t25 + t654 - t656) * t45 - 2 * (-t89 * t1 + t510 * t25) * t6 * t11 * t4 - 2 * t72 * t1) * t385
    t716 = t47 * t22
    t718 = t12 * t63
    drd53 = ((-2 * t716 - 2 * t718 - t55 + t47) * t45 + t6 * (t18 * t25 * t22 + (-t27 - t29 + t30) * t1 - 2 * t302) * t4 + t80 * t1) * t385
    drd61 = drd43
    drd62 = drd53
    drd63 = 2 * t4 * t7 * ((-t25 * t1 + t26) * t4 + t81) * t500
    t743 = t273 - t396 - Uy + ty
    dsd11 = ((2 * t501 - 2 * t521 - 2 * t507) * t4 + 2 * t6 * (t743 * t25 * t1 + t11 * t315 - t226 + t420 - t512)) * t271
    t755 = (t27 - t29 - t26) * t25
    dsd12 = (2 * t543 * t4 - 2 * t6 * (t533 + (t535 - t41 - t66) * t1 - t755)) * t385
    dsd13 = 2 * t7 * (-t564 * t4 + t59) * t385
    t765 = t17 + 1
    t766 = t765 * t2
    dsd21 = ((t766 * t90 + (t92 + t93 + t54) * t25 * t1 - t99 - t101 + t39) * t4 + t6 * (t14 * t8 * t1 + t104 + t113)) * t271
    dsd22 = ((t766 * t48 + (-t274 - t279 - t280) * t1 + (t284 + t285 - t676 + t677) * t25) * t4 - t79 * t72 * t6) * t385
    dsd23 = ((-2 * t765 * t8 * t48 + (-t391 - t199 - t392) * t1 + t514) * t4 - t10 * t6 * t564) * t385
    dsd31 = ((t533 + ((t243 - t244 + t245 + t177) * t17 - t41 - t369) * t1 + 2 * t755) * t4 - t6 * (t523 + (t55 + t632) * t25 * t1 - 2 * t8 * t253)) * t271
    dsd32 = ((-t512 - 2 * (t273 + t396 / 2 - Uy + ty) * t25 * t1 - 2 * t612 * t11 * t17 - t199 + 2 * t11 * t66) * t4 + 2 * t6 * (-t501 / 2 + (t10 * t66 + t410) * t1 - (t84 + t686) * t25 / 2)) * t385
    dsd33 = -((t404 + t25 * t687 * t1 - t169 - 2 * t10 * (t207 - t209 + t205 - t136 + t210 + t206)) * t4 - t650 * t6) * t7 * t500
    dsd41 = (2 * t743 * t1 * t4 - 2 * (-t7 * t2 + t84 * t22 - t252) * t6) * t271
    t860 = (t7 - 1) * (t7 + 1)
    dsd42 = ((-2 * t510 * t119 - t860 * t63 + t26 - t27 + t29) * t4 - t6 * ((-2 * t55 - t47) * t1 + t86 * t11)) * t500
    dsd43 = t7 * ((t716 + t718 + t55 + t47) * t4 + t6 * t698) * t500
    dsd51 = dsd42
    dsd52 = (2 * (t860 * t25 + t510 * t3) * t11 * t4 - 2 * t6 * t1 * t87) * t385
    dsd53 = t7 * ((t684 + (t84 - t252) * t1 - 2 * t86) * t4 - t79 * t6 * t1) * t385
    dsd61 = dsd43
    dsd62 = dsd53
    dsd63 = -2 * t79 * t4 * t17 * t500


    dres_dU = np.zeros((2*N, 3))
    dres_dU[::2, :] = np.array([drdUx, drdUy, drdUz]).T
    dres_dU[1::2, :] = np.array([dsdUx, dsdUy, dsdUz]).T
    Hr = np.array([
        [drd11, drd12, drd13],
        [drd21, drd22, drd23],
        [drd31, drd32, drd33],
        [drd41, drd42, drd43],
        [drd51, drd52, drd53],
        [drd61, drd62, drd63],
    ])
    Hr = np.moveaxis(Hr, -1, 0) # Last axis is N-dimensional. Put it the first.
    Hs = np.array([
        [dsd11, dsd12, dsd13],
        [dsd21, dsd22, dsd23],
        [dsd31, dsd32, dsd33],
        [dsd41, dsd42, dsd43],
        [dsd51, dsd52, dsd53],
        [dsd61, dsd62, dsd63],
    ])
    Hs = np.moveaxis(Hs, -1, 0) # Last axis is N-dimensional. Put it the first.
    ddU_dres_dx = np.zeros((2*N, 6, 3))
    ddU_dres_dx[::2, :, :] = Hr
    ddU_dres_dx[1::2, :, :] = Hs
    return dres_dU, ddU_dres_dx