"""Bootstrap a wall-level CRT alignment fit AUGMENTED with one extra
per-wall deformation parameter.

Three modes (priorities #1, #2, #4 from the alignment-extension menu):

  --mode bow    out-of-plane wall sag/bow.
                Adds 1 param b_w (cm) per wall.  For a space-point at
                world position P assigned to wall w (centroid C_w,
                normal n̂_w), apply
                    δP = b_w · ( |r_⊥| / r0 )² · n̂_w
                where r_⊥ = (P - C_w) projected into the wall plane and
                r0 = 200 cm.  b_w is the bow at r=200 cm.

  --mode shear  in-plane shear (proxy for per-module inter-layer
                rotation, here aggregated to the wall).
                Adds 1 param α_w (rad) per wall.
                    δP = α_w · ((P - C_w)·ê1_w) · ê2_w
                with (ê1, ê2) the two in-plane axes of the wall.
                Captures a rotation between the wall's two strip layers.

  --mode scale  in-plane isotropic scale.
                Adds 1 param s_w (dimensionless) per wall.
                    δP = s_w · ( (P - C_w) - ((P - C_w)·n̂)·n̂ )
                Wall stretch / shrink.  Partly degenerate with global
                gauge; we keep the TPC0-anode pin to fix that.

Wall normals by tagger ID (verified from align_df sample centroids):
  0  Bot     n̂=ŷ
  1  South   n̂=ẑ
  2  North   n̂=ẑ
  3  West    n̂=x̂
  4  East    n̂=x̂
  5  TopLow  n̂=ŷ
  6  TopHigh n̂=ŷ

Layout of the parameter vector (n_walls = 7):
    [n_walls × 3 dT, n_walls × 3 dR,
     3 anode1_dT,    3 anode1_dR,
     n_walls × 1 deformation]
total = 7·6 + 6 + 7 = 55 params.

Output (in --outdir):
  central_theta.npy
  bootstrap_thetas.npy            shape (B, 55)
  bootstrap_summary.csv           per-parameter mean / std / quantiles
  bootstrap_param_index.csv
  deformation_summary.csv         per-wall central + std + pull for the
                                  new deformation parameter (THE thing
                                  we want to read).

Usage:
  python bootstrap_wall_deformation.py --mode bow   --n-boot 1000
  python bootstrap_wall_deformation.py --mode shear --n-boot 1000
  python bootstrap_wall_deformation.py --mode scale --n-boot 1000
"""

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool, cpu_count

try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from align_crt import (
    load_align_df, explode_to_residual_rows, build_module_index,
    pack_inputs_for_jax, load_gdml_centroids,
    ANODE_CENTROID, ANODE_X_NOMINAL,
)


from crtlib.constants import WALL_NORMAL_AXIS as _WALL_NORMAL_AXIS, TAG_NAME
from crtlib.wall import basis_per_wall as _basis_per_wall


def _make_weighted_loss(sp1, sp2, m1i, m2i, ap_nom, atpc,
                        wall_centroids, anode_centroids,
                        wall_normals, wall_e1, wall_e2,
                        n_walls, sigma_cm,
                        prior_dT_cm, prior_dR_rad, prior_def,
                        mode, r0_cm=200.0,
                        fractional=False, frac_clamp_cm=1.0):
    """JIT-compiled weighted loss with per-wall deformation augment."""
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    jax.config.update("jax_enable_x64", True)

    sigma2 = sigma_cm ** 2
    inv_sT2 = 1.0 / (prior_dT_cm ** 2)
    inv_sR2 = 1.0 / (prior_dR_rad ** 2)
    inv_sD2 = 1.0 / (prior_def ** 2)
    mc = jnp.asarray(wall_centroids)
    ac = jnp.asarray(anode_centroids)
    nh = jnp.asarray(wall_normals)
    e1 = jnp.asarray(wall_e1)
    e2 = jnp.asarray(wall_e2)
    ZERO3 = jnp.zeros(3, dtype=jnp.float64)
    sp1_j = jnp.asarray(sp1, dtype=jnp.float64)
    sp2_j = jnp.asarray(sp2, dtype=jnp.float64)
    m1i_j = jnp.asarray(m1i, dtype=jnp.int32)
    m2i_j = jnp.asarray(m2i, dtype=jnp.int32)
    ap_j  = jnp.asarray(ap_nom, dtype=jnp.float64)
    atpc_j = jnp.asarray(atpc, dtype=jnp.int32)
    n_w = n_walls
    inv_r0_2 = 1.0 / (r0_cm ** 2)

    if fractional:
        line0 = sp2_j - sp1_j
        u0 = line0 / (jnp.linalg.norm(line0, axis=1, keepdims=True) + 1e-9)
        delta0 = ap_j - sp1_j
        proj0 = jnp.sum(delta0 * u0, axis=1, keepdims=True) * u0
        d0_2 = jnp.sum((delta0 - proj0) ** 2, axis=1)
        inv_d0_2 = 1.0 / (d0_2 + frac_clamp_cm ** 2)
    else:
        inv_d0_2 = None

    def unpack(theta):
        idx = 0
        mod_dT = theta[idx:idx + n_w * 3].reshape(n_w, 3); idx += n_w * 3
        mod_dR = theta[idx:idx + n_w * 3].reshape(n_w, 3); idx += n_w * 3
        a1_dT = theta[idx:idx + 3]; idx += 3
        a1_dR = theta[idx:idx + 3]; idx += 3
        deform = theta[idx:idx + n_w]; idx += n_w
        anode_dT = jnp.stack([ZERO3, a1_dT], axis=0)
        anode_dR = jnp.stack([ZERO3, a1_dR], axis=0)
        return mod_dT, mod_dR, anode_dT, anode_dR, deform

    def rodrigues(dR_per_pt, v):
        ang2 = jnp.sum(dR_per_pt ** 2, axis=1, keepdims=True)
        small = ang2 < 1e-10
        safe_ang2 = jnp.where(small, 1.0, ang2)
        safe_ang = jnp.sqrt(safe_ang2)
        s = jnp.where(small, 1.0 - ang2 / 6.0, jnp.sin(safe_ang) / safe_ang)
        c = jnp.where(small, 0.5 - ang2 / 24.0, (1.0 - jnp.cos(safe_ang)) / safe_ang2)
        cross_tv = jnp.cross(dR_per_pt, v)
        return v + s * cross_tv + c * jnp.cross(dR_per_pt, cross_tv)

    def deform_points(P, wall_idx, deform_per_wall):
        """Add the per-wall deformation correction to P (in nominal frame)."""
        c = mc[wall_idx]                 # (N, 3) wall centroid per row
        n = nh[wall_idx]                 # (N, 3) wall normal
        v = P - c                        # (N, 3) offset from centroid
        if mode == "bow":
            v_norm = jnp.sum(v * n, axis=1, keepdims=True)
            v_perp = v - v_norm * n      # in-plane offset
            r2 = jnp.sum(v_perp ** 2, axis=1, keepdims=True)
            return P + deform_per_wall[wall_idx, None] * (r2 * inv_r0_2) * n
        if mode == "shear":
            ev1 = e1[wall_idx]
            ev2 = e2[wall_idx]
            u = jnp.sum(v * ev1, axis=1, keepdims=True)
            return P + deform_per_wall[wall_idx, None] * u * ev2
        if mode == "scale":
            v_norm = jnp.sum(v * n, axis=1, keepdims=True)
            v_perp = v - v_norm * n
            return P + deform_per_wall[wall_idx, None] * v_perp
        raise ValueError(f"unknown mode {mode}")

    def transform(P, c, dT, dR):
        v = P - c
        return rodrigues(dR, v) + c + dT

    def loss(theta, weights):
        mod_dT, mod_dR, anode_dT, anode_dR, deform = unpack(theta)
        # 1. Apply the per-wall deformation IN THE NOMINAL FRAME.
        P1d = deform_points(sp1_j, m1i_j, deform)
        P2d = deform_points(sp2_j, m2i_j, deform)
        # 2. Then the rigid 6-DOF wall transform.
        c1 = mc[m1i_j]; c2 = mc[m2i_j]
        P1 = transform(P1d, c1, mod_dT[m1i_j], mod_dR[m1i_j])
        P2 = transform(P2d, c2, mod_dT[m2i_j], mod_dR[m2i_j])
        # 3. Anode.
        ca = ac[atpc_j]
        Pa = transform(ap_j, ca, anode_dT[atpc_j], anode_dR[atpc_j])
        # 4. Straightness residual.
        line = P2 - P1
        u = line / (jnp.linalg.norm(line, axis=1, keepdims=True) + 1e-9)
        delta = Pa - P1
        proj = jnp.sum(delta * u, axis=1, keepdims=True) * u
        d2 = jnp.sum((delta - proj) ** 2, axis=1)
        if fractional:
            d2 = d2 * inv_d0_2
        chi2 = jnp.sum(weights * d2) / sigma2
        # 5. Priors.
        reg_T = inv_sT2 * (jnp.sum(mod_dT ** 2) + jnp.sum(anode_dT[1] ** 2))
        reg_R = inv_sR2 * (jnp.sum(mod_dR ** 2) + jnp.sum(anode_dR[1] ** 2))
        reg_D = inv_sD2 * jnp.sum(deform ** 2)
        return chi2 + reg_T + reg_R + reg_D

    return jit(loss), jit(grad(loss))


# ---------- worker state ----------
_LOSS = None
_GRAD = None
_INIT_THETA = None
_N_RESIDUALS = None
_OPTS = None


def _worker_init(payload):
    global _LOSS, _GRAD, _INIT_THETA, _N_RESIDUALS, _OPTS
    sp1, sp2, m1i, m2i, ap_nom, atpc = payload["inputs"]
    _LOSS, _GRAD = _make_weighted_loss(
        sp1, sp2, m1i, m2i, ap_nom, atpc,
        payload["wall_centroids"], payload["anode_centroids"],
        payload["wall_normals"], payload["wall_e1"], payload["wall_e2"],
        payload["n_walls"], payload["sigma_cm"],
        payload["prior_dT_cm"], payload["prior_dR_rad"], payload["prior_def"],
        payload["mode"],
        fractional=payload.get("fractional", False),
        frac_clamp_cm=payload.get("frac_clamp_cm", 1.0),
    )
    _INIT_THETA = payload["init_theta"]
    _N_RESIDUALS = payload["n_residuals"]
    _OPTS = payload["lbfgs_opts"]
    import jax.numpy as jnp
    w = jnp.ones(_N_RESIDUALS, dtype=jnp.float64)
    th = jnp.asarray(_INIT_THETA, dtype=jnp.float64)
    _ = float(_LOSS(th, w))


def _central_fit(_=None):
    import jax.numpy as jnp
    from scipy.optimize import minimize
    weights_one = jnp.ones(_N_RESIDUALS, dtype=jnp.float64)
    def f_and_g(theta_np):
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        l = float(_LOSS(theta, weights_one))
        g = np.asarray(_GRAD(theta, weights_one), dtype=np.float64)
        return l, g
    res = minimize(f_and_g, _INIT_THETA, jac=True, method="L-BFGS-B",
                   options=_OPTS)
    return res.x


def _bootstrap_one(seed):
    import jax.numpy as jnp
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(_N_RESIDUALS,
                              np.full(_N_RESIDUALS, 1.0 / _N_RESIDUALS))
    weights = weights.astype(np.float64)
    weights_j = jnp.asarray(weights)
    def f_and_g(theta_np):
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        l = float(_LOSS(theta, weights_j))
        g = np.asarray(_GRAD(theta, weights_j), dtype=np.float64)
        return l, g
    res = minimize(f_and_g, _INIT_THETA, jac=True, method="L-BFGS-B",
                   options=_OPTS)
    return res.x


def _build_inputs(parquet_path, cut_config, max_pre_residual_cm, min_counts,
                  init_dz_cm, init_dz_tags,
                  centroid_source, gdml_centroids_csv):
    df = load_align_df(parquet_path)
    with open(cut_config) as f:
        cuts = json.load(f)
    tmin_lo = float(cuts["tpc_t_min_lo"])
    tmin_hi = float(cuts["tpc_t_min_hi"])

    rows = explode_to_residual_rows(df, tmin_lo, tmin_hi, min_crt_len_cm=0.0)

    s1 = rows[["crt_sp1_x", "crt_sp1_y", "crt_sp1_z"]].to_numpy()
    s2 = rows[["crt_sp2_x", "crt_sp2_y", "crt_sp2_z"]].to_numpy()
    pa = np.stack([rows["anode_x_nom"].to_numpy(),
                   rows["anode_y"].to_numpy(),
                   rows["anode_z"].to_numpy()], axis=1)
    line = s2 - s1
    L = np.linalg.norm(line, axis=1, keepdims=True)
    u = line / np.maximum(L, 1e-9)
    delta = pa - s1
    proj = np.sum(delta * u, axis=1, keepdims=True) * u
    pre_resid = np.linalg.norm(delta - proj, axis=1)
    rows = rows.loc[pre_resid <= max_pre_residual_cm].reset_index(drop=True)

    mod_ids, mod_to_idx, centroids, counts, taggers = build_module_index(rows, df)
    keep_mods = mod_ids[counts >= min_counts]
    keep_mask = (rows["mod1_id"].isin(keep_mods) & rows["mod2_id"].isin(keep_mods))
    rows = rows[keep_mask].reset_index(drop=True)
    mod_ids, mod_to_idx, centroids, counts, taggers = build_module_index(rows, df)

    sp1, sp2, m1i_mod, m2i_mod, ap_nom, atpc = pack_inputs_for_jax(rows, mod_to_idx)

    if centroid_source == "gdml":
        print(f"  using GDML centroids from {gdml_centroids_csv}", flush=True)
        centroids = load_gdml_centroids(mod_ids, centroids,
                                        gdml_csv=gdml_centroids_csv)

    # Aggregate to walls.
    unique_walls = np.array(sorted(set(int(t) for t in taggers)),
                            dtype=np.int32)
    wall_to_idx = {int(w): i for i, w in enumerate(unique_walls)}
    mod_to_wall = np.array([wall_to_idx[int(t)] for t in taggers],
                           dtype=np.int32)
    m1i = mod_to_wall[m1i_mod]
    m2i = mod_to_wall[m2i_mod]
    n_walls = len(unique_walls)
    wall_centroids = np.zeros((n_walls, 3))
    for i, w in enumerate(unique_walls):
        mask = (taggers == w)
        wall_centroids[i] = centroids[mask].mean(axis=0)
    wall_normals, wall_e1, wall_e2 = _basis_per_wall(unique_walls)

    n_params = n_walls * 6 + 6 + n_walls
    init_theta = np.zeros(n_params, dtype=np.float64)
    if init_dz_cm and init_dz_tags:
        init_tags = set(int(t) for t in str(init_dz_tags).split(",") if t)
        for i, t in enumerate(unique_walls):
            if int(t) in init_tags:
                init_theta[i * 3 + 2] = init_dz_cm

    return {
        "inputs": (np.asarray(sp1), np.asarray(sp2), m1i, m2i,
                   np.asarray(ap_nom), atpc),
        "wall_centroids": wall_centroids,
        "anode_centroids": np.array([ANODE_CENTROID[0], ANODE_CENTROID[1]]),
        "wall_normals": wall_normals,
        "wall_e1": wall_e1,
        "wall_e2": wall_e2,
        "n_walls": n_walls,
        "n_residuals": len(rows),
        "init_theta": init_theta,
        "wall_taggers": unique_walls,
    }


_TAG_NAME = TAG_NAME  # alias for backwards-compat with existing references


def _param_label(i, n_walls, wall_taggers, mode):
    if i < n_walls * 3:
        widx = i // 3
        comp = ["dx", "dy", "dz"][i % 3]
        return f"{_TAG_NAME[int(wall_taggers[widx])]}_{comp}"
    if i < n_walls * 6:
        ii = i - n_walls * 3
        widx = ii // 3
        comp = ["dthx", "dthy", "dthz"][ii % 3]
        return f"{_TAG_NAME[int(wall_taggers[widx])]}_{comp}"
    j = i - n_walls * 6
    if j < 3:
        return f"anode1_{['dx','dy','dz'][j]}"
    if j < 6:
        return f"anode1_{['dthx','dthy','dthz'][j-3]}"
    widx = j - 6
    suffix = {"bow": "bow_cm", "shear": "shear_rad",
              "scale": "scale"}[mode]
    return f"{_TAG_NAME[int(wall_taggers[widx])]}_{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("bow", "shear", "scale"))
    ap.add_argument("--parquet",
                    default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--outdir", default=None,
                    help="default: bootstrap_wall_<mode>_gdml")
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--max-pre-residual-cm", type=float, default=50.0)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--init-dz-cm", type=float, default=-10.0)
    ap.add_argument("--init-dz-tags", type=str, default="5,6")
    ap.add_argument("--sigma-cm", type=float, default=5.0)
    ap.add_argument("--prior-dT-cm", type=float, default=5.0)
    ap.add_argument("--prior-dR-rad", type=float, default=0.01)
    ap.add_argument("--prior-bow-cm", type=float, default=5.0,
                    help="prior 1-sigma on per-wall bow at r0=200 cm")
    ap.add_argument("--prior-shear-rad", type=float, default=0.05,
                    help="prior 1-sigma on per-wall in-plane shear")
    ap.add_argument("--prior-scale", type=float, default=0.05,
                    help="prior 1-sigma on per-wall in-plane fractional scale")
    ap.add_argument("--fractional", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="DEFAULT ON. Minimise Σ(d_i / d_0_i)² instead of "
                         "Σ d_i². Pass --no-fractional for legacy absolute.")
    ap.add_argument("--frac-clamp-cm", type=float, default=1.0,
                    help="floor on d_0_i (cm) when --fractional is on")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260509)
    ap.add_argument("--centroid-source", choices=("empirical", "gdml"),
                    default="gdml")
    ap.add_argument("--gdml-centroids-csv",
                    default="crt_module_gdml_centroids.csv")
    args = ap.parse_args()

    if args.outdir is None:
        args.outdir = f"bootstrap_wall_{args.mode}_gdml"
    os.makedirs(args.outdir, exist_ok=True)

    prior_def = {"bow": args.prior_bow_cm, "shear": args.prior_shear_rad,
                 "scale": args.prior_scale}[args.mode]

    print(f"Mode: {args.mode}   prior_def: {prior_def}", flush=True)

    print(f"Building inputs ...", flush=True)
    payload = _build_inputs(
        args.parquet, args.cut_config, args.max_pre_residual_cm,
        args.min_counts, args.init_dz_cm, args.init_dz_tags,
        centroid_source=args.centroid_source,
        gdml_centroids_csv=args.gdml_centroids_csv,
    )
    payload["sigma_cm"] = args.sigma_cm
    payload["prior_dT_cm"] = args.prior_dT_cm
    payload["prior_dR_rad"] = args.prior_dR_rad
    payload["prior_def"] = prior_def
    payload["fractional"] = bool(args.fractional)
    payload["frac_clamp_cm"] = float(args.frac_clamp_cm)
    payload["mode"] = args.mode
    payload["lbfgs_opts"] = {"maxiter": args.max_iter, "ftol": 1e-12,
                             "gtol": 1e-9, "disp": False}
    n_walls = payload["n_walls"]
    n_params = n_walls * 6 + 6 + n_walls
    print(f"  N residuals : {payload['n_residuals']:,}", flush=True)
    print(f"  n_walls     : {n_walls}  taggers = {payload['wall_taggers'].tolist()}",
          flush=True)
    print(f"  parameters  : {n_params}  (incl. {n_walls} deformation)",
          flush=True)
    print(f"  init dz     : {args.init_dz_cm} on tags {args.init_dz_tags}",
          flush=True)

    # Central fit.
    print("\nCentral fit (full sample) ...", flush=True)
    t0 = time.time()
    with Pool(processes=1, initializer=_worker_init,
              initargs=(payload,)) as pool:
        theta_central = pool.apply(_central_fit)
    print(f"  central fit time = {time.time()-t0:.1f}s", flush=True)
    np.save(os.path.join(args.outdir, "central_theta.npy"), theta_central)

    # Bootstrap.
    print(f"\nBootstrap: B = {args.n_boot}, nproc = {args.nproc} ...",
          flush=True)
    nproc = min(args.nproc, args.n_boot, cpu_count())
    seeds = list(range(args.seed, args.seed + args.n_boot))

    t0 = time.time()
    with Pool(processes=nproc, initializer=_worker_init,
              initargs=(payload,)) as pool:
        results = []
        for k, x in enumerate(pool.imap_unordered(_bootstrap_one, seeds)):
            results.append(x)
            if (k + 1) % max(1, args.n_boot // 10) == 0:
                print(f"  {k+1}/{args.n_boot}  "
                      f"({(k+1)/(time.time()-t0):.1f} fits/s)", flush=True)
    elapsed = time.time() - t0
    print(f"  total bootstrap time: {elapsed:.1f}s "
          f"({args.n_boot/elapsed:.2f} fits/s)", flush=True)

    thetas = np.array(results)
    np.save(os.path.join(args.outdir, "bootstrap_thetas.npy"), thetas)

    # Per-parameter summary table + index.
    rows = []
    index_rows = []
    for i in range(n_params):
        label = _param_label(i, n_walls, payload["wall_taggers"], args.mode)
        v = thetas[:, i]
        rows.append({
            "param": label,
            "central":     float(theta_central[i]),
            "boot_mean":   float(np.mean(v)),
            "boot_median": float(np.median(v)),
            "boot_std":    float(np.std(v, ddof=1)),
            "boot_p16":    float(np.percentile(v, 16)),
            "boot_p84":    float(np.percentile(v, 84)),
        })
        if i < n_walls * 3:
            widx = i // 3
            comp = ["dx", "dy", "dz"][i % 3]
            tag = int(payload["wall_taggers"][widx])
            kind = "wall"
        elif i < n_walls * 6:
            ii = i - n_walls * 3
            widx = ii // 3
            comp = ["dthx", "dthy", "dthz"][ii % 3]
            tag = int(payload["wall_taggers"][widx])
            kind = "wall"
        elif i < n_walls * 6 + 3:
            comp = ["dx", "dy", "dz"][i - n_walls * 6]
            tag = -1; kind = "anode"
        elif i < n_walls * 6 + 6:
            comp = ["dthx", "dthy", "dthz"][i - n_walls * 6 - 3]
            tag = -1; kind = "anode"
        else:
            widx = i - n_walls * 6 - 6
            comp = {"bow": "bow_cm", "shear": "shear_rad",
                    "scale": "scale"}[args.mode]
            tag = int(payload["wall_taggers"][widx])
            kind = "deformation"
        index_rows.append({
            "param_idx": i, "param": label, "tagger": tag,
            "component": comp, "kind": kind,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.outdir, "bootstrap_summary.csv"),
                   index=False)
    pd.DataFrame(index_rows).to_csv(
        os.path.join(args.outdir, "bootstrap_param_index.csv"), index=False)

    # Deformation-only summary table — the headline result.
    def_rows = []
    for w in range(n_walls):
        i = n_walls * 6 + 6 + w
        v = thetas[:, i]
        cen = float(theta_central[i])
        std = float(np.std(v, ddof=1))
        pull = cen / std if std > 0 else float("nan")
        tag = int(payload["wall_taggers"][w])
        def_rows.append({
            "wall_id": tag,
            "wall_name": _TAG_NAME[tag],
            "mode": args.mode,
            "central": cen,
            "boot_mean": float(np.mean(v)),
            "boot_std": std,
            "boot_p16": float(np.percentile(v, 16)),
            "boot_p84": float(np.percentile(v, 84)),
            "pull": pull,
            "abs_pull": abs(pull),
            "significant_3sigma": bool(abs(pull) > 3.0),
        })
    def_df = pd.DataFrame(def_rows).sort_values("abs_pull", ascending=False)
    def_df.to_csv(os.path.join(args.outdir, "deformation_summary.csv"),
                  index=False)

    print(f"\n=== {args.mode.upper()} deformation per wall  "
          f"(B = {len(thetas)}, |pull|>3 ⇒ significant) ===",
          flush=True)
    print(f"{'wall':<10s}  {'central':>10s}  {'boot_mean':>10s}  "
          f"{'boot_std':>10s}  {'pull':>7s}", flush=True)
    for _, r in def_df.iterrows():
        flag = "  ***" if r["significant_3sigma"] else ""
        print(f"{r['wall_name']:<10s}  {r['central']:+10.5f}  "
              f"{r['boot_mean']:+10.5f}  {r['boot_std']:10.5f}  "
              f"{r['pull']:+7.2f}{flag}", flush=True)

    # Deformation-distribution plots: 7 walls in one figure.
    fig, axes = plt.subplots(1, n_walls,
                             figsize=(2.5 * n_walls, 3.4),
                             sharey=True)
    for w, ax in enumerate(axes):
        i = n_walls * 6 + 6 + w
        v = thetas[:, i]
        cen = theta_central[i]
        ax.hist(v, bins=30, histtype="step", linewidth=1.4, color="C0")
        ax.axvline(cen, color="C3", linestyle="--", linewidth=1.0,
                   label="central")
        ax.axvline(0.0, color="0.5", linestyle=":", linewidth=0.8)
        std = float(np.std(v, ddof=1))
        pull = cen / std if std > 0 else 0.0
        ax.set_title(f"{_TAG_NAME[int(payload['wall_taggers'][w])]}\n"
                     f"{cen:+.4f} ± {std:.4f}  pull={pull:+.2f}",
                     fontsize=10)
    fig.suptitle(f"Per-wall {args.mode} (B={len(thetas)}, "
                 f"prior σ={prior_def})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"deformation_{args.mode}.{ext}"),
                    dpi=130)
    plt.close(fig)

    print(f"\n  wrote {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
