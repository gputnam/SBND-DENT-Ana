"""JAX rigid-body kinematics + alignment-loss factories.

Two top-level entry points:

  ``make_module_loss(...)``   per-module 6-DOF fit (used by align_crt.py
                              in module mode; param vector length
                              ``6*n_modules + 6``).

  ``make_wall_loss(...)``     per-wall 6-DOF fit OPTIONALLY augmented
                              with one of three per-wall deformation
                              modes (``bow``, ``shear``, ``scale``).
                              Param vector length:
                                ``6*n_walls + 6  + (n_walls if mode else 0)``.

The math is shared:
  - axis-angle (Rodrigues) rotations with the smooth-at-θ=0 "double
    where" idiom so gradients are well-defined at the origin
  - bootstrap-weighted χ² (set ``weights=None`` for the unweighted fit)
  - separate Tikhonov priors on translations vs rotations vs deformation

The TPC0 anode is held FIXED (its pose is the global gauge); only TPC1
gets 6 DOF. The CRT-side parameters are organised per-module or
per-wall depending on the entry point.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit

# 64-bit precision is necessary for the cross-product / projection math
# at module scales (cm) with sub-mm alignment shifts.
jax.config.update("jax_enable_x64", True)

ZERO3 = jnp.zeros(3, dtype=jnp.float64)


def rodrigues_rotate(dR_per_pt, v):
    """Apply axis-angle rotation `dR` (length=angle, dir=axis) to vector v.

    Vectorized over the leading axis. Smooth (gradient-safe at θ=0)
    via the "double where" idiom: route ang2 through `where(small,1,ang2)`
    BEFORE every operation that might divide by it, so the dead branch
    never produces 0/0 even in the gradient pass.
    """
    ang2 = jnp.sum(dR_per_pt ** 2, axis=1, keepdims=True)
    small = ang2 < 1e-10
    safe_ang2 = jnp.where(small, 1.0, ang2)
    safe_ang = jnp.sqrt(safe_ang2)
    s_full = jnp.sin(safe_ang) / safe_ang
    c_full = (1.0 - jnp.cos(safe_ang)) / safe_ang2
    s_taylor = 1.0 - ang2 / 6.0
    c_taylor = 0.5 - ang2 / 24.0
    s = jnp.where(small, s_taylor, s_full)
    c = jnp.where(small, c_taylor, c_full)
    cross_tv = jnp.cross(dR_per_pt, v)
    return v + s * cross_tv + c * jnp.cross(dR_per_pt, cross_tv)


def transform_points(P_nominal, centroid_per_pt, dT_per_pt, dR_per_pt):
    """Rigid-body transform: rotate-about-centroid, then translate."""
    v = P_nominal - centroid_per_pt
    return rodrigues_rotate(dR_per_pt, v) + centroid_per_pt + dT_per_pt


def perp_distance_sq(sp1, sp2, anode_pt):
    """Squared perpendicular distance of `anode_pt` from line through (sp1, sp2)."""
    line = sp2 - sp1
    u = line / (jnp.linalg.norm(line, axis=1, keepdims=True) + 1e-9)
    delta = anode_pt - sp1
    proj = jnp.sum(delta * u, axis=1, keepdims=True) * u
    perp = delta - proj
    return jnp.sum(perp ** 2, axis=1)


# =======================================================================
# Per-module loss
# =======================================================================

def make_module_loss(sp1, sp2, mod1_idx, mod2_idx, anode_p_nom, anode_tpc,
                     module_centroids, anode_centroids,
                     n_modules, sigma_cm,
                     prior_dT_cm, prior_dR_rad,
                     weighted=False,
                     fractional=False, frac_clamp_cm=1.0):
    """Return ``(loss_fn, unpack_fn)``. If `weighted=True` the loss takes
    ``(theta, weights)`` else just ``theta``.

    Param vector layout: ``[n_mod·3 dT, n_mod·3 dR, 3 anode1_dT, 3 anode1_dR]``.

    fractional :
        if True the per-residual contribution becomes ``(d_i / d_0_i)^2``
        where ``d_0_i`` is the **θ=0 (pre-fit, nominal-frame)**
        perpendicular distance for residual i. Clamped from below by
        ``frac_clamp_cm`` so already-near-zero residuals don't dominate.
        Useful when the goal is to reduce residuals MULTIPLICATIVELY
        rather than absolutely.
    """
    n_mod = n_modules
    sigma2 = sigma_cm ** 2
    inv_sT2 = 1.0 / (prior_dT_cm ** 2)
    inv_sR2 = 1.0 / (prior_dR_rad ** 2)
    mc = jnp.asarray(module_centroids)
    ac = jnp.asarray(anode_centroids)

    if fractional:
        d0_2 = perp_distance_sq(jnp.asarray(sp1, dtype=jnp.float64),
                                jnp.asarray(sp2, dtype=jnp.float64),
                                jnp.asarray(anode_p_nom, dtype=jnp.float64))
        inv_d0_2 = 1.0 / (d0_2 + frac_clamp_cm ** 2)
    else:
        inv_d0_2 = None

    def unpack(theta):
        idx = 0
        mod_dT = theta[idx:idx + n_mod * 3].reshape(n_mod, 3); idx += n_mod * 3
        mod_dR = theta[idx:idx + n_mod * 3].reshape(n_mod, 3); idx += n_mod * 3
        a1_dT = theta[idx:idx + 3]; idx += 3
        a1_dR = theta[idx:idx + 3]; idx += 3
        anode_dT = jnp.stack([ZERO3, a1_dT], axis=0)
        anode_dR = jnp.stack([ZERO3, a1_dR], axis=0)
        return mod_dT, mod_dR, anode_dT, anode_dR

    def _loss_inner(theta, weights):
        mod_dT, mod_dR, anode_dT, anode_dR = unpack(theta)
        c1 = mc[mod1_idx]; c2 = mc[mod2_idx]
        P1 = transform_points(sp1, c1, mod_dT[mod1_idx], mod_dR[mod1_idx])
        P2 = transform_points(sp2, c2, mod_dT[mod2_idx], mod_dR[mod2_idx])
        ca = ac[anode_tpc]
        Pa = transform_points(anode_p_nom, ca,
                              anode_dT[anode_tpc], anode_dR[anode_tpc])
        d2 = perp_distance_sq(P1, P2, Pa)
        if fractional:
            d2 = d2 * inv_d0_2
        if weights is None:
            chi2 = jnp.sum(d2) / sigma2
        else:
            chi2 = jnp.sum(weights * d2) / sigma2
        reg_T = inv_sT2 * (jnp.sum(mod_dT ** 2) + jnp.sum(anode_dT[1] ** 2))
        reg_R = inv_sR2 * (jnp.sum(mod_dR ** 2) + jnp.sum(anode_dR[1] ** 2))
        return chi2 + reg_T + reg_R

    if weighted:
        def loss(theta, weights):
            return _loss_inner(theta, weights)
        return jit(loss), unpack
    else:
        def loss(theta):
            return _loss_inner(theta, None)
        return jit(loss), unpack


# =======================================================================
# Per-wall loss (with optional deformation augmentation)
# =======================================================================

def make_wall_loss(sp1, sp2, w1i, w2i, anode_p_nom, anode_tpc,
                   wall_centroids, anode_centroids,
                   wall_normals, wall_e1, wall_e2,
                   n_walls, sigma_cm,
                   prior_dT_cm, prior_dR_rad,
                   prior_deform=None, deform_mode=None, r0_cm=200.0,
                   weighted=True,
                   fractional=False, frac_clamp_cm=1.0):
    """Return ``(loss_fn, unpack_fn, n_params)``. Loss takes ``(theta, weights)``.

    Pass ``deform_mode=None`` for vanilla 6-DOF wall fit (48 params for
    7 walls). Pass ``"bow" | "shear" | "scale"`` for the augmented fit.

    Param vector layout::

        [n_w·3 dT, n_w·3 dR, 3 anode1_dT, 3 anode1_dR, (n_w deform)]

    fractional :
        if True the per-residual contribution becomes ``(d_i / d_0_i)^2``
        where ``d_0_i`` is the **θ=0 (pre-fit, nominal-frame)**
        perpendicular distance for residual i. Clamped from below by
        ``frac_clamp_cm``.
    """
    sigma2 = sigma_cm ** 2
    inv_sT2 = 1.0 / (prior_dT_cm ** 2)
    inv_sR2 = 1.0 / (prior_dR_rad ** 2)
    n_w = n_walls
    has_deform = deform_mode is not None
    if has_deform:
        assert prior_deform is not None
        inv_sD2 = 1.0 / (prior_deform ** 2)
    inv_r0_2 = 1.0 / (r0_cm ** 2)

    if fractional:
        d0_2 = perp_distance_sq(jnp.asarray(sp1, dtype=jnp.float64),
                                jnp.asarray(sp2, dtype=jnp.float64),
                                jnp.asarray(anode_p_nom, dtype=jnp.float64))
        inv_d0_2 = 1.0 / (d0_2 + frac_clamp_cm ** 2)
    else:
        inv_d0_2 = None

    mc = jnp.asarray(wall_centroids)
    ac = jnp.asarray(anode_centroids)
    nh = jnp.asarray(wall_normals)
    e1 = jnp.asarray(wall_e1)
    e2 = jnp.asarray(wall_e2)

    n_params = n_w * 6 + 6 + (n_w if has_deform else 0)

    def unpack(theta):
        idx = 0
        w_dT = theta[idx:idx + n_w * 3].reshape(n_w, 3); idx += n_w * 3
        w_dR = theta[idx:idx + n_w * 3].reshape(n_w, 3); idx += n_w * 3
        a1_dT = theta[idx:idx + 3]; idx += 3
        a1_dR = theta[idx:idx + 3]; idx += 3
        anode_dT = jnp.stack([ZERO3, a1_dT], axis=0)
        anode_dR = jnp.stack([ZERO3, a1_dR], axis=0)
        if has_deform:
            deform = theta[idx:idx + n_w]
            return w_dT, w_dR, anode_dT, anode_dR, deform
        return w_dT, w_dR, anode_dT, anode_dR, None

    def deform_points(P, wall_idx, deform_per_wall):
        c = mc[wall_idx]; n = nh[wall_idx]; v = P - c
        if deform_mode == "bow":
            v_norm = jnp.sum(v * n, axis=1, keepdims=True)
            v_perp = v - v_norm * n
            r2 = jnp.sum(v_perp ** 2, axis=1, keepdims=True)
            return P + deform_per_wall[wall_idx, None] * (r2 * inv_r0_2) * n
        if deform_mode == "shear":
            ev1 = e1[wall_idx]; ev2 = e2[wall_idx]
            u = jnp.sum(v * ev1, axis=1, keepdims=True)
            return P + deform_per_wall[wall_idx, None] * u * ev2
        if deform_mode == "scale":
            v_norm = jnp.sum(v * n, axis=1, keepdims=True)
            v_perp = v - v_norm * n
            return P + deform_per_wall[wall_idx, None] * v_perp
        return P  # mode=None handled in caller

    def loss(theta, weights):
        w_dT, w_dR, anode_dT, anode_dR, deform = unpack(theta)
        if has_deform:
            P1d = deform_points(sp1, w1i, deform)
            P2d = deform_points(sp2, w2i, deform)
        else:
            P1d, P2d = sp1, sp2
        c1 = mc[w1i]; c2 = mc[w2i]
        P1 = transform_points(P1d, c1, w_dT[w1i], w_dR[w1i])
        P2 = transform_points(P2d, c2, w_dT[w2i], w_dR[w2i])
        ca = ac[anode_tpc]
        Pa = transform_points(anode_p_nom, ca,
                              anode_dT[anode_tpc], anode_dR[anode_tpc])
        d2 = perp_distance_sq(P1, P2, Pa)
        if fractional:
            d2 = d2 * inv_d0_2
        chi2 = jnp.sum(weights * d2) / sigma2
        reg_T = inv_sT2 * (jnp.sum(w_dT ** 2) + jnp.sum(anode_dT[1] ** 2))
        reg_R = inv_sR2 * (jnp.sum(w_dR ** 2) + jnp.sum(anode_dR[1] ** 2))
        out = chi2 + reg_T + reg_R
        if has_deform:
            out = out + inv_sD2 * jnp.sum(deform ** 2)
        return out

    return jit(loss), jit(grad(loss)), unpack, n_params
