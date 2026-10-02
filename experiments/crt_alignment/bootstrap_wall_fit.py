"""Bootstrap the wall-level OR module-level CRT alignment fit to
obtain per-parameter uncertainties.

Approach: bootstrap-by-weights.
  Each replicate draws a multinomial weight vector w ∈ ℕ^N with N
  observations from N residual rows (so Σwᵢ = N). The loss is then
      chi2 = Σᵢ wᵢ · dᵢ² / σ²
  and the same JIT-compiled loss/grad pair is reused for every
  replicate.

Modes:
  default (per-module): 105 modules × 6 DOF + 6 anode DOF = 636 params
  --by-wall:            7 walls × 6 DOF + 6 anode DOF = 48 params

Multiprocessing:
  Each worker JIT-compiles the loss function once in its initializer,
  then handles many bootstrap replicates from the pool's task queue.

Output (in --outdir):
  bootstrap_thetas.npy        shape (B, n_params), one row per replicate
  bootstrap_summary.csv       per-parameter mean / std / quantiles
  bootstrap_param_index.csv   parameter ID → human label
                              (FEB MAC for module mode, tagger ID for
                              wall mode)
  central_theta.npy           full-sample fit (no resampling)
  bootstrap_pulls.{png,pdf}   histograms

Usage:
  python bootstrap_wall_fit.py [--n-boot 1000] [--nproc 40] [--by-wall]
"""

import argparse
import json
import multiprocessing as _mp
import os
import time
import warnings
from multiprocessing import Pool, cpu_count

# Force the "spawn" start method BEFORE any matplotlib / numpy / pandas
# imports below would otherwise initialize thread pools. Without this,
# Pool(processes=N).apply() forks while the parent has dozens of internal
# threads (multiprocessing managers, numpy OpenMP, matplotlib backend,
# etc.), and the worker deadlocks before it can import JAX — the
# "JAX is multithreaded, so this will likely lead to a deadlock"
# RuntimeWarning is the parent's diagnostic, but the actual hang is from
# inherited locks in the worker. "spawn" starts each worker in a fresh
# interpreter so no parent state is inherited; payload is pickled.
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Lower priority so we don't crowd the host. Must be set BEFORE forking.
try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

# Restrict each worker process to a single CPU thread so that 40 workers
# don't oversubscribe a 64-core host. JAX/XLA reads these env vars.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

warnings.simplefilter(action="ignore", category=FutureWarning)

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse main fit's plumbing.
from align_crt import (
    load_align_df,
    explode_to_residual_rows,
    build_module_index,
    pack_inputs_for_jax,
    load_gdml_centroids,
    ANODE_CENTROID,
    ANODE_X_NOMINAL,
)


# ---------- jax loss with bootstrap weights ----------
# Keep this module-level so it can be imported in workers via the
# initializer.

def _make_weighted_loss(sp1, sp2, m1i, m2i, ap_nom, atpc,
                        centroids, anode_centroids,
                        n_units, sigma_cm, prior_dT_cm, prior_dR_rad,
                        fractional=False, frac_clamp_cm=1.0):
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    jax.config.update("jax_enable_x64", True)

    sigma2 = sigma_cm ** 2
    inv_sT2 = 1.0 / (prior_dT_cm ** 2)
    inv_sR2 = 1.0 / (prior_dR_rad ** 2)
    mc = jnp.asarray(centroids)
    ac = jnp.asarray(anode_centroids)
    ZERO3 = jnp.zeros(3, dtype=jnp.float64)
    sp1_j = jnp.asarray(sp1, dtype=jnp.float64)
    sp2_j = jnp.asarray(sp2, dtype=jnp.float64)
    m1i_j = jnp.asarray(m1i, dtype=jnp.int32)
    m2i_j = jnp.asarray(m2i, dtype=jnp.int32)
    ap_j  = jnp.asarray(ap_nom, dtype=jnp.float64)
    atpc_j = jnp.asarray(atpc, dtype=jnp.int32)
    n_w = n_units

    if fractional:
        # θ=0 perpendicular distance per residual (in nominal frame).
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
        anode_dT = jnp.stack([ZERO3, a1_dT], axis=0)
        anode_dR = jnp.stack([ZERO3, a1_dR], axis=0)
        return mod_dT, mod_dR, anode_dT, anode_dR

    def rodrigues(dR_per_pt, v):
        ang2 = jnp.sum(dR_per_pt ** 2, axis=1, keepdims=True)
        small = ang2 < 1e-10
        safe_ang2 = jnp.where(small, 1.0, ang2)
        safe_ang = jnp.sqrt(safe_ang2)
        s = jnp.where(small, 1.0 - ang2 / 6.0, jnp.sin(safe_ang) / safe_ang)
        c = jnp.where(small, 0.5 - ang2 / 24.0, (1.0 - jnp.cos(safe_ang)) / safe_ang2)
        cross_tv = jnp.cross(dR_per_pt, v)
        return v + s * cross_tv + c * jnp.cross(dR_per_pt, cross_tv)

    def transform(P, c, dT, dR):
        v = P - c
        return rodrigues(dR, v) + c + dT

    def loss(theta, weights):
        mod_dT, mod_dR, anode_dT, anode_dR = unpack(theta)
        c1 = mc[m1i_j]; c2 = mc[m2i_j]
        P1 = transform(sp1_j, c1, mod_dT[m1i_j], mod_dR[m1i_j])
        P2 = transform(sp2_j, c2, mod_dT[m2i_j], mod_dR[m2i_j])
        ca = ac[atpc_j]
        Pa = transform(ap_j, ca, anode_dT[atpc_j], anode_dR[atpc_j])
        line = P2 - P1
        u = line / (jnp.linalg.norm(line, axis=1, keepdims=True) + 1e-9)
        delta = Pa - P1
        proj = jnp.sum(delta * u, axis=1, keepdims=True) * u
        d2 = jnp.sum((delta - proj) ** 2, axis=1)
        if fractional:
            d2 = d2 * inv_d0_2
        chi2 = jnp.sum(weights * d2) / sigma2
        reg_T = inv_sT2 * (jnp.sum(mod_dT ** 2) + jnp.sum(anode_dT[1] ** 2))
        reg_R = inv_sR2 * (jnp.sum(mod_dR ** 2) + jnp.sum(anode_dR[1] ** 2))
        return chi2 + reg_T + reg_R

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
        payload["centroids"], payload["anode_centroids"],
        payload["n_units"], payload["sigma_cm"],
        payload["prior_dT_cm"], payload["prior_dR_rad"],
        fractional=payload.get("fractional", False),
        frac_clamp_cm=payload.get("frac_clamp_cm", 1.0),
    )
    _INIT_THETA = payload["init_theta"]
    _N_RESIDUALS = payload["n_residuals"]
    _OPTS = payload["lbfgs_opts"]
    # Warm-start the JIT — pay compilation now, before fitting.
    import jax.numpy as jnp
    w = jnp.ones(_N_RESIDUALS, dtype=jnp.float64)
    th = jnp.asarray(_INIT_THETA, dtype=jnp.float64)
    _ = float(_LOSS(th, w))


def _central_fit(_=None):
    """Full-sample fit (weights all = 1). Runs inside a worker so JAX
    never initializes in the parent process. Returns the optimized theta."""
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
    """Fit one bootstrap replicate. Returns the optimized theta."""
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


# ---------- main ----------
def _build_inputs(parquet_path, cut_config, max_pre_residual_cm, min_counts,
                  init_dz_cm, init_dz_tags, by_wall=False,
                  centroid_source="empirical",
                  gdml_centroids_csv="crt_module_gdml_centroids.csv",
                  val_frac=0.1, val_seed=42):
    """Reproduce the residual-row preparation + module/wall index build
    used in align_crt.py, returning everything needed to run a fit.

    by_wall = True: 7 wall-level alignment units.
    by_wall = False (default): per-module alignment units (~105).
    """
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

    if by_wall:
        unique_walls = np.array(sorted(set(int(t) for t in taggers)),
                                dtype=np.int32)
        wall_to_idx = {int(w): i for i, w in enumerate(unique_walls)}
        mod_to_wall = np.array([wall_to_idx[int(t)] for t in taggers],
                               dtype=np.int32)
        m1i = mod_to_wall[m1i_mod]
        m2i = mod_to_wall[m2i_mod]
        n_units = len(unique_walls)
        wall_centroids = np.zeros((n_units, 3))
        for i, w in enumerate(unique_walls):
            mask = (taggers == w)
            wall_centroids[i] = centroids[mask].mean(axis=0)
        unit_centroids = wall_centroids
        unit_ids = unique_walls
        unit_label = "wall"
        unit_taggers = unique_walls  # one per unit
    else:
        m1i = m1i_mod
        m2i = m2i_mod
        n_units = len(mod_ids)
        unit_centroids = centroids
        unit_ids = mod_ids
        unit_label = "module"
        unit_taggers = taggers  # one per unit (the module's tagger)

    n_params = n_units * 6 + 6
    init_theta = np.zeros(n_params, dtype=np.float64)
    if init_dz_cm and init_dz_tags:
        init_tags = set(int(t) for t in str(init_dz_tags).split(",") if t)
        for i, t in enumerate(unit_taggers):
            if int(t) in init_tags:
                init_theta[i * 3 + 2] = init_dz_cm

    # 90/10 train/val split (consistent with align_crt). The bootstrap
    # only resamples on TRAIN rows; val is held out for residual plots.
    n_total_rows = len(rows)
    rng_split = np.random.default_rng(val_seed)
    is_val = rng_split.random(n_total_rows) < val_frac
    val_idx = np.where(is_val)[0]
    train_idx = np.where(~is_val)[0]
    sp1_t = np.asarray(sp1)[train_idx]; sp2_t = np.asarray(sp2)[train_idx]
    m1i_t = m1i[train_idx]; m2i_t = m2i[train_idx]
    ap_t = np.asarray(ap_nom)[train_idx]; atpc_t = atpc[train_idx]
    print(f"  train/val split: {len(train_idx):,} train, "
          f"{len(val_idx):,} val (val_frac={val_frac}, seed={val_seed})",
          flush=True)

    return {
        "inputs": (sp1_t, sp2_t, m1i_t, m2i_t, ap_t, atpc_t),
        "inputs_full": (np.asarray(sp1), np.asarray(sp2), m1i, m2i,
                        np.asarray(ap_nom), atpc),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "centroids": unit_centroids,
        "anode_centroids": np.array([ANODE_CENTROID[0], ANODE_CENTROID[1]]),
        "n_units": n_units,
        "n_residuals": len(train_idx),
        "n_residuals_full": len(rows),
        "init_theta": init_theta,
        "unit_ids": unit_ids,
        "unit_taggers": np.asarray(unit_taggers, dtype=np.int32),
        "unit_label": unit_label,
        "n_per_unit": counts if not by_wall else None,
    }


def _param_label(i, unit_ids, unit_label):
    """Human-readable label for parameter index i."""
    n_u = len(unit_ids)
    if i < n_u * 3:
        uidx = i // 3
        comp = ["dx", "dy", "dz"][i % 3]
        return f"{unit_label}{int(unit_ids[uidx])}_{comp}"
    if i < n_u * 6:
        uidx = (i - n_u * 3) // 3
        comp = ["dthx", "dthy", "dthz"][(i - n_u * 3) % 3]
        return f"{unit_label}{int(unit_ids[uidx])}_{comp}"
    j = i - n_u * 6
    if j < 3:
        return f"anode1_{['dx','dy','dz'][j]}"
    return f"anode1_{['dthx','dthy','dthz'][j-3]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--outdir", default="bootstrap_wall")
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--max-pre-residual-cm", type=float, default=50.0)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--init-dz-cm", type=float, default=-10.0)
    ap.add_argument("--init-dz-tags", type=str, default="5,6")
    ap.add_argument("--sigma-cm", type=float, default=5.0)
    ap.add_argument("--prior-dT-cm", type=float, default=5.0)
    ap.add_argument("--prior-dR-rad", type=float, default=0.01)
    ap.add_argument("--fractional", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="DEFAULT ON. Minimise Σ(d_i / d_0_i)² instead of "
                         "Σ d_i². Pass --no-fractional for legacy absolute.")
    ap.add_argument("--frac-clamp-cm", type=float, default=1.0,
                    help="floor on d_0_i (cm) when --fractional is on")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20250508)
    ap.add_argument("--by-wall", action="store_true",
                    help="Bootstrap the wall-level fit (7 walls × 6 DOF + 6) "
                         "instead of the per-module fit (N_modules × 6 + 6).")
    ap.add_argument("--centroid-source", choices=("empirical", "gdml"),
                    default="empirical")
    ap.add_argument("--gdml-centroids-csv",
                    default="crt_module_gdml_centroids.csv")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="hold out this fraction of residuals for validation "
                         "(default 0.1 = 90/10 train/val split)")
    ap.add_argument("--val-seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Building inputs (mode = {'wall' if args.by_wall else 'module'}) ...",
          flush=True)
    payload = _build_inputs(
        args.parquet, args.cut_config, args.max_pre_residual_cm,
        args.min_counts, args.init_dz_cm, args.init_dz_tags,
        by_wall=args.by_wall,
        centroid_source=args.centroid_source,
        gdml_centroids_csv=args.gdml_centroids_csv,
        val_frac=args.val_frac, val_seed=args.val_seed,
    )
    payload["sigma_cm"] = args.sigma_cm
    payload["prior_dT_cm"] = args.prior_dT_cm
    payload["prior_dR_rad"] = args.prior_dR_rad
    payload["fractional"] = bool(args.fractional)
    payload["frac_clamp_cm"] = float(args.frac_clamp_cm)
    payload["lbfgs_opts"] = {"maxiter": args.max_iter, "ftol": 1e-12,
                             "gtol": 1e-9, "disp": False}
    print(f"  N residuals : {payload['n_residuals']:,}", flush=True)
    print(f"  unit_label  : {payload['unit_label']}", flush=True)
    print(f"  n_units     : {payload['n_units']}", flush=True)
    print(f"  parameters  : {payload['n_units']*6 + 6}", flush=True)
    print(f"  init dz     : {args.init_dz_cm} on tags {args.init_dz_tags}",
          flush=True)

    # Central full-sample fit (weights = 1). Run it in a single-worker
    # pool so JAX never gets initialized in the parent process — that
    # avoids the JAX-state-doesn't-survive-fork pathology that hangs
    # bootstrap workers.
    print("\nCentral fit (full sample) ...", flush=True)
    t0 = time.time()
    with Pool(processes=1, initializer=_worker_init,
              initargs=(payload,)) as pool:
        theta_central = pool.apply(_central_fit)
    print(f"  central fit time = {time.time()-t0:.1f}s", flush=True)
    np.save(os.path.join(args.outdir, "central_theta.npy"), theta_central)

    # Bootstrap pool — fresh fork, JAX state is built only inside workers.
    print(f"\nBootstrap: B = {args.n_boot}, nproc = {args.nproc} ...", flush=True)
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

    thetas = np.array(results)  # (B, n_params)
    np.save(os.path.join(args.outdir, "bootstrap_thetas.npy"), thetas)

    # Per-parameter summary + index.
    n_params = thetas.shape[1]
    rows = []
    index_rows = []
    n_u = payload["n_units"]
    for i in range(n_params):
        label = _param_label(i, payload["unit_ids"], payload["unit_label"])
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
        # Map parameter index → unit_id, tagger, component, kind.
        if i < n_u * 3:
            uidx = i // 3
            comp = ["dx", "dy", "dz"][i % 3]
            uid = int(payload["unit_ids"][uidx])
            tag = int(payload["unit_taggers"][uidx])
            kind = payload["unit_label"]
        elif i < n_u * 6:
            ii = i - n_u * 3
            uidx = ii // 3
            comp = ["dthx", "dthy", "dthz"][ii % 3]
            uid = int(payload["unit_ids"][uidx])
            tag = int(payload["unit_taggers"][uidx])
            kind = payload["unit_label"]
        elif i < n_u * 6 + 3:
            comp = ["dx", "dy", "dz"][i - n_u * 6]
            uid = 1; tag = -1; kind = "anode"
        else:
            comp = ["dthx", "dthy", "dthz"][i - n_u * 6 - 3]
            uid = 1; tag = -1; kind = "anode"
        index_rows.append({
            "param_idx": i, "param": label,
            "unit_id": uid, "tagger": tag,
            "component": comp, "kind": kind,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.outdir, "bootstrap_summary.csv"),
                   index=False)
    pd.DataFrame(index_rows).to_csv(
        os.path.join(args.outdir, "bootstrap_param_index.csv"), index=False)

    # Pretty-print.
    print(f"\nPer-parameter bootstrap summary "
          f"(B={len(thetas)}, n_residuals={payload['n_residuals']:,}):", flush=True)
    print(f"{'param':<14s}  {'central':>10s}  {'boot_mean':>10s}  {'boot_std':>10s}  "
          f"{'p16':>10s}  {'p84':>10s}", flush=True)
    for _, r in summary.iterrows():
        print(f"{r['param']:<14s}  {r['central']:+10.4f}  {r['boot_mean']:+10.4f}  "
              f"{r['boot_std']:10.4f}  {r['boot_p16']:+10.4f}  {r['boot_p84']:+10.4f}",
              flush=True)

    # Per-unit pull histograms — only useful in wall mode (n_units ≲ 10).
    # In module mode, defer to the dedicated intra-wall outlier analysis.
    if n_u <= 12:
        fig, axes = plt.subplots(n_u + 1, 6, figsize=(18, 2.6 * (n_u + 1)))
        if n_u == 0:
            axes = np.atleast_2d(axes)
        comps = ["dx", "dy", "dz", "dθx", "dθy", "dθz"]
        for r in range(n_u):
            for c in range(6):
                ax = axes[r, c]
                i = r * 3 + c if c < 3 else n_u * 3 + r * 3 + (c - 3)
                v = thetas[:, i]
                cen = theta_central[i]
                ax.hist(v, bins=30, histtype="step", linewidth=1.3, color="C0")
                ax.axvline(cen, color="C3", linestyle="--", linewidth=0.8)
                ax.set_title(f"{payload['unit_label']}{int(payload['unit_ids'][r])} {comps[c]}",
                             fontsize=9)
                ax.set_xlabel(f"{np.std(v, ddof=1):.3f} ± std", fontsize=7)
        for c in range(6):
            ax = axes[-1, c]
            i = n_u * 6 + c if c < 3 else n_u * 6 + 3 + (c - 3)
            v = thetas[:, i]
            cen = theta_central[i]
            ax.hist(v, bins=30, histtype="step", linewidth=1.3, color="C2")
            ax.axvline(cen, color="C3", linestyle="--", linewidth=0.8)
            ax.set_title(f"TPC1-anode {comps[c]}", fontsize=9)
            ax.set_xlabel(f"{np.std(v, ddof=1):.3f} ± std", fontsize=7)
        fig.suptitle(f"Bootstrap distributions (B={len(thetas)}, "
                     f"unit={payload['unit_label']})", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.outdir, f"bootstrap_pulls.{ext}"),
                        dpi=110)
        plt.close(fig)

    print(f"\n  results in {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
