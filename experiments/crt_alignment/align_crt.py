"""Per-module CRT alignment via straightness of anode-piercing CRT-matched
TPC tracks.

For every selected track we have three (or four) co-linear points:
  - one or two anode-piercing points (TPC0 anode, TPC1 anode, or both
    for anode-cathode crossers)
  - two CRT space-points (start and end of the matched CRT track),
    each known to belong to a specific CRT module (FEB MAC5 ID)

The free parameters are 6-DOF rigid-body corrections per CRT module
(translation + small-angle rotation about the module's nominal
centroid) and per the TPC1 anode (same parameterization about its
nominal anode-plane centroid). The TPC0 anode is held FIXED at its
nominal pose — that fixes the global rigid-body gauge, since global
translations/rotations of the whole detector would move TPC0's anode.
With the gauge fixed by the TPC0 anode, no soft penalty is needed and
none is applied. The straightness metric is the perpendicular distance
of the anode point from the line through the two (corrected) CRT
points. Total loss is sum-of-squares of those distances plus weak
Tikhonov regularization on the free corrections.

Outputs (in --outdir, default ./crt_alignment/):
  crt_alignment_results.csv  — per-module + per-anode correction values
  crt_alignment_results.json — the same, plus loss curve and metadata
  hist_residual_pre_post.{png,pdf}
  per_module_shifts.{png,pdf}
  per_anode_shifts.{png,pdf}

Usage:
  python align_crt.py /exp/sbnd/data/users/gputnam/efield-study/align_df
"""

import argparse
import glob
import json
import os
import time

DEFAULT_CUT_CONFIG = "cut_tuning/cut_config_per_tpc.json"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize

# 64-bit precision is necessary for the cross-product / projection math
# at module scales (cm) with sub-mm alignment shifts.
jax.config.update("jax_enable_x64", True)

# Constants and helpers live in crtlib so that all scripts share one
# source of truth. We re-export them here for backwards compatibility
# (older callers do `from align_crt import ANODE_X_NOMINAL` etc.).
from crtlib.constants import ANODE_X_NOMINAL, ANODE_CENTROID
from crtlib.io import load_align_df
from crtlib.residuals import explode_to_residual_rows as _explode_canonical
from crtlib.align import (build_module_index, load_gdml_centroids,
                          pack_inputs_for_jax)
from crtlib import jax_loss as _jax_loss


def explode_to_residual_rows(df, tmin_lo, tmin_hi, min_crt_len_cm=400.0):
    """Backwards-compatible alias for `crtlib.residuals.explode_to_residual_rows`."""
    return _explode_canonical(df, tmin_lo, tmin_hi, min_crt_len_cm)


def make_loss(sp1, sp2, mod1_idx, mod2_idx, anode_p_nom, anode_tpc,
              module_centroids, anode_centroids,
              n_modules, sigma_cm,
              prior_dT_cm, prior_dR_rad,
              fractional=False, frac_clamp_cm=1.0):
    """Backwards-compatible per-module loss (delegates to
    `crtlib.jax_loss.make_module_loss`).

    Param vector length ``6*n_modules + 6``: per-module 6-DOF +
    TPC1 anode 6-DOF. TPC0 anode is fixed (gauge).

    fractional :
        if True, minimise Σ(d_i / d_0_i)² instead of Σ d_i², where
        d_0_i is the θ=0 perpendicular distance for residual i, with
        a `frac_clamp_cm` floor.
    """
    return _jax_loss.make_module_loss(
        sp1, sp2, mod1_idx, mod2_idx, anode_p_nom, anode_tpc,
        module_centroids, anode_centroids,
        n_modules, sigma_cm,
        prior_dT_cm, prior_dR_rad,
        weighted=False,
        fractional=fractional, frac_clamp_cm=frac_clamp_cm,
    )


def run_optimization(loss_fn, n_params, *, max_iter=500, theta0=None,
                     bounds=None):
    grad_fn = jit(grad(loss_fn))

    def f_and_g(theta_np):
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        l = float(loss_fn(theta))
        g = np.asarray(grad_fn(theta), dtype=np.float64)
        return l, g

    if theta0 is None:
        theta0 = np.zeros(n_params, dtype=np.float64)
    else:
        theta0 = np.asarray(theta0, dtype=np.float64)
        assert theta0.shape == (n_params,)
    history = [float(loss_fn(jnp.asarray(theta0, dtype=jnp.float64)))]

    def cb(xk):
        history.append(float(loss_fn(jnp.asarray(xk, dtype=jnp.float64))))

    print(f"  starting L-BFGS-B with {n_params} parameters", flush=True)
    print(f"  initial loss (at theta0): {history[0]:.4f}", flush=True)
    if bounds is not None:
        n_pinned = sum(1 for b in bounds if b == (0.0, 0.0))
        print(f"  bounds active: {n_pinned} parameters pinned at 0", flush=True)
    t0 = time.time()
    res = minimize(f_and_g, theta0, jac=True, method="L-BFGS-B",
                   callback=cb, bounds=bounds,
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9,
                            "disp": False})
    print(f"  L-BFGS-B finished in {time.time() - t0:.1f}s, success={res.success}, "
          f"nit={res.nit}, final loss={res.fun:.4f}", flush=True)
    return res.x, history, res


def write_results(theta, mod_ids, mod_counts, history, outdir, kind="module",
                  centroids=None, anode_centroids=None):
    n_mod = len(mod_ids)
    mod_dT = theta[:n_mod * 3].reshape(n_mod, 3)
    mod_dR = theta[n_mod * 3:n_mod * 6].reshape(n_mod, 3)
    anode1_dT = theta[n_mod * 6:n_mod * 6 + 3]
    anode1_dR = theta[n_mod * 6 + 3:n_mod * 6 + 6]
    anode_dT = np.stack([np.zeros(3), anode1_dT], axis=0)
    anode_dR = np.stack([np.zeros(3), anode1_dR], axis=0)

    rows = []
    for i, mac in enumerate(mod_ids):
        cx = cy = cz = float("nan")
        if centroids is not None:
            cx, cy, cz = (float(centroids[i, 0]), float(centroids[i, 1]),
                          float(centroids[i, 2]))
        rows.append({
            "kind": kind, "module_id": int(mac), "n_residuals": int(mod_counts[i]),
            "centroid_x": cx, "centroid_y": cy, "centroid_z": cz,
            "dx": float(mod_dT[i, 0]), "dy": float(mod_dT[i, 1]), "dz": float(mod_dT[i, 2]),
            "dthx": float(mod_dR[i, 0]), "dthy": float(mod_dR[i, 1]), "dthz": float(mod_dR[i, 2]),
        })
    for k in (0, 1):
        cx = cy = cz = float("nan")
        if anode_centroids is not None:
            cx, cy, cz = (float(anode_centroids[k, 0]),
                          float(anode_centroids[k, 1]),
                          float(anode_centroids[k, 2]))
        rows.append({
            "kind": "anode", "module_id": k,
            "n_residuals": -2 if k == 0 else -1,  # -2 = pinned (gauge), -1 = fit
            "centroid_x": cx, "centroid_y": cy, "centroid_z": cz,
            "dx": float(anode_dT[k, 0]), "dy": float(anode_dT[k, 1]), "dz": float(anode_dT[k, 2]),
            "dthx": float(anode_dR[k, 0]), "dthy": float(anode_dR[k, 1]), "dthz": float(anode_dR[k, 2]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "crt_alignment_results.csv"), index=False)

    payload = {
        "results": rows,
        "loss_history": history,
        "n_modules": n_mod,
    }
    with open(os.path.join(outdir, "crt_alignment_results.json"), "w") as fp:
        json.dump(payload, fp, indent=2)


def make_plots(theta_init, theta_final, residuals_init, residuals_final,
               mod_ids, mod_counts, history, outdir):
    n_mod = len(mod_ids)

    # 1) Residual distribution before and after.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, np.percentile(residuals_init, 99.5), 100)
    ax.hist(residuals_init, bins=bins, histtype="step", linewidth=1.5,
            label=f"pre  (median {np.median(residuals_init):.2f} cm)")
    ax.hist(residuals_final, bins=bins, histtype="step", linewidth=1.5,
            label=f"post (median {np.median(residuals_final):.2f} cm)")
    ax.set_xlabel("perpendicular residual (cm)")
    ax.set_ylabel("residuals / bin")
    ax.set_yscale("log")
    ax.set_title("Anode-to-CRT-line straightness residual")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"hist_residual_pre_post.{ext}"), dpi=150)
    plt.close(fig)

    # 2) Per-module shift magnitude (translation, rotation).
    mod_dT = theta_final[:n_mod * 3].reshape(n_mod, 3)
    mod_dR = theta_final[n_mod * 3:n_mod * 6].reshape(n_mod, 3)
    dT_mag = np.linalg.norm(mod_dT, axis=1)
    dR_mag = np.linalg.norm(mod_dR, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(dT_mag, bins=50, histtype="step", linewidth=1.5)
    axes[0].set_xlabel("|Δposition| (cm)")
    axes[0].set_ylabel("modules")
    axes[0].set_title("Per-module translation magnitudes")
    axes[1].hist(dR_mag * 1e3, bins=50, histtype="step", linewidth=1.5)
    axes[1].set_xlabel("|δθ| (mrad)")
    axes[1].set_title("Per-module rotation magnitudes")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"per_module_shifts.{ext}"), dpi=150)
    plt.close(fig)

    # 3) Loss curve.
    if history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history)
        ax.set_yscale("log")
        ax.set_xlabel("L-BFGS-B iteration")
        ax.set_ylabel("loss")
        ax.set_title("Optimization loss curve")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"loss_curve.{ext}"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input",
                    help="align_df parquet or directory of shards")
    ap.add_argument("--outdir", default="crt_alignment")
    ap.add_argument("--sigma-cm", type=float, default=5.0,
                    help="per-residual gaussian width (CRT spatial resolution; cm)")
    ap.add_argument("--prior-dT-cm", type=float, default=5.0,
                    help="prior 1-sigma on per-module/per-anode translation "
                         "magnitude (cm). Smaller = stronger constraint.")
    ap.add_argument("--prior-dR-rad", type=float, default=0.05,
                    help="prior 1-sigma on per-module/per-anode rotation "
                         "magnitude (rad). 50 mrad ~ 3°.")
    ap.add_argument("--fractional", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="DEFAULT ON. Minimise Σ(d_i / d_0_i)² (relative "
                         "residuals) instead of Σ d_i². d_0_i is the θ=0 "
                         "perpendicular distance for residual i. Pass "
                         "--no-fractional for the legacy absolute-residual "
                         "loss.")
    ap.add_argument("--frac-clamp-cm", type=float, default=1.0,
                    help="floor on d_0_i (cm) for the fractional loss; "
                         "residuals with d_0_i below this don't dominate.")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="hold out this fraction of residuals for validation "
                         "(default 0.1 = 90/10 train/val split). Pre/post "
                         "residuals are saved separately for both samples.")
    ap.add_argument("--val-seed", type=int, default=42,
                    help="seed for the train/val random split")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--centroid-source", choices=("empirical", "gdml"),
                    default="empirical",
                    help="Per-module rotation pivot: empirical = mean of "
                         "CRT space-points attributed to each module (cosmic-"
                         "flux-biased); gdml = matched v02_06 GDML world "
                         "centroids (geometry-defined).")
    ap.add_argument("--gdml-centroids-csv",
                    default="crt_module_gdml_centroids.csv")
    ap.add_argument("--no-rotations", action="store_true",
                    help="Pin all rotation DOFs to zero — only fit "
                         "translations. Useful as a baseline to see "
                         "how much of the alignment comes from rotation.")
    ap.add_argument("--by-wall", action="store_true",
                    help="Fit ONE 6-DOF correction per CRT WALL (tagger) "
                         "instead of per module. Reduces n_params from "
                         "~6×n_mod to 6×n_walls (~42). Useful as a baseline "
                         "and to isolate wall-coherent systematics.")
    ap.add_argument("--init-dz-cm", type=float, default=0.0,
                    help="Initial-guess z-shift (cm) applied to modules whose "
                         "tagger ID is in --init-dz-tags. Use to absorb a "
                         "wall-coherent z-bias before per-module fitting.")
    ap.add_argument("--init-dz-tags", type=str, default="5,6",
                    help="Comma-separated tagger IDs to receive --init-dz-cm "
                         "as their starting z-correction (default: 5,6 = "
                         "TopLow + TopHigh).")
    ap.add_argument("--cut-config", default=DEFAULT_CUT_CONFIG,
                    help="cut_tuning/cut_config.json — provides the tight "
                         "in-time-anode window used to gate per-TPC anode "
                         "points.")
    ap.add_argument("--min-crt-len-cm", type=float, default=400.0,
                    help="reject CRT-tracks with sp2-sp1 length below this. "
                         "Most TopHigh+TopLow-only matches have ~150 cm of "
                         "lever arm and produce huge extrapolation errors.")
    ap.add_argument("--max-pre-residual-cm", type=float, default=100.0,
                    help="Drop residual rows whose PRE-alignment "
                         "perpendicular distance exceeds this. Per-tagger "
                         "diagnostics show tracks with TopLow / TopHigh "
                         "CRT hits often have ~200 cm residuals — those "
                         "look like wall-level systematics or genuine "
                         "match errors that per-module 6-DOF alignment "
                         "can't recover from. Default 100 cm keeps the "
                         "well-behaved bulk and rejects the obvious "
                         "outliers before fitting.")
    ap.add_argument("--min-counts", type=int, default=10,
                    help="drop modules with fewer than this many CRT-hits")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.input} ...", flush=True)
    df = load_align_df(args.input)
    print(f"  align_df: {len(df):,} rows", flush=True)
    print(f"  anode-cathode: {int(df['is_anode_cathode_crosser'].sum()):,}", flush=True)
    print(f"  in-time anode: {int(df['is_intime_anode_crosser'].sum()):,}", flush=True)

    with open(args.cut_config) as f:
        cuts = json.load(f)
    # New unified cut: hit times in TrackCaloSkim are already
    # CRT-track-t0-corrected (whicht0=1), so the per-TPC earliest hit
    # tick equals the "CRT-shifted" anode-piercing test directly.
    if "tpc_t_min_lo" in cuts:
        tmin_lo = float(cuts["tpc_t_min_lo"])
        tmin_hi = float(cuts["tpc_t_min_hi"])
    else:  # legacy envelope-based config
        tmin_lo = float(cuts["intime_anode_crosser"]["t_min_p2_lo"])
        tmin_hi = float(cuts["intime_anode_crosser"]["t_min_p2_hi"])
    print(f"  per-TPC anode-piercing gate: t_min in [{tmin_lo}, {tmin_hi}] ticks",
          flush=True)

    print("Exploding to residual rows ...", flush=True)
    rows = explode_to_residual_rows(df, tmin_lo, tmin_hi,
                                    min_crt_len_cm=args.min_crt_len_cm)
    print(f"  residual rows (one per actual anode-piercing, "
          f"crt_len > {args.min_crt_len_cm:.0f} cm): {len(rows):,}",
          flush=True)

    # Drop rows whose pre-alignment perpendicular residual is > threshold.
    # These are dominated by tracks that traverse top-CRT walls only or
    # have CRT-TPC mis-matches; per-module 6-DOF alignment can't fix them
    # and they only bias the bulk fit. Computed in pure numpy.
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
    keep = pre_resid <= args.max_pre_residual_cm
    print(f"  pre-alignment perp residual (cm) "
          f"P10/P50/P90 = {np.percentile(pre_resid, [10,50,90])}", flush=True)
    print(f"  keeping {int(keep.sum()):,} of {len(rows):,} rows "
          f"(pre-residual <= {args.max_pre_residual_cm} cm)", flush=True)
    rows = rows.loc[keep].reset_index(drop=True)

    print("Building module index ...", flush=True)
    mod_ids, mod_to_idx, centroids, counts, taggers = build_module_index(rows, df)
    print(f"  unique modules: {len(mod_ids)}", flush=True)
    print(f"  count distribution (per module):")
    print(f"    min/median/max = {counts.min()}/{int(np.median(counts))}/{counts.max()}", flush=True)
    sparse = mod_ids[counts < args.min_counts]
    if len(sparse):
        print(f"  modules below --min-counts={args.min_counts}: "
              f"{len(sparse)} (will be dropped)", flush=True)
        keep_mask = ((rows["mod1_id"].isin(mod_ids[counts >= args.min_counts])) &
                     (rows["mod2_id"].isin(mod_ids[counts >= args.min_counts])))
        rows = rows[keep_mask].reset_index(drop=True)
        mod_ids, mod_to_idx, centroids, counts, taggers = build_module_index(rows, df)
        print(f"  retained: {len(rows):,} residual rows, {len(mod_ids)} modules",
              flush=True)

    sp1_full, sp2_full, m1i_mod_full, m2i_mod_full, ap_nom_full, atpc_full = (
        pack_inputs_for_jax(rows, mod_to_idx))
    anode_centroids = np.array([ANODE_CENTROID[0], ANODE_CENTROID[1]])

    if args.centroid_source == "gdml":
        print(f"  using GDML centroids from {args.gdml_centroids_csv}",
              flush=True)
        centroids = load_gdml_centroids(mod_ids, centroids,
                                        gdml_csv=args.gdml_centroids_csv)

    # ---------- 90 / 10 train / val split ----------
    n_total_rows = len(rows)
    rng_split = np.random.default_rng(args.val_seed)
    is_val = rng_split.random(n_total_rows) < args.val_frac
    val_idx = np.where(is_val)[0]
    train_idx = np.where(~is_val)[0]
    print(f"  train/val split: {len(train_idx):,} train, "
          f"{len(val_idx):,} val (val_frac={args.val_frac}, "
          f"seed={args.val_seed})", flush=True)
    np.save(os.path.join(args.outdir, "train_idx.npy"), train_idx)
    np.save(os.path.join(args.outdir, "val_idx.npy"), val_idx)

    sp1 = sp1_full[train_idx]; sp2 = sp2_full[train_idx]
    m1i_mod = m1i_mod_full[train_idx]; m2i_mod = m2i_mod_full[train_idx]
    ap_nom = ap_nom_full[train_idx]; atpc = atpc_full[train_idx]

    if args.by_wall:
        # Wall-level fit: each unique tagger gets ONE shared 6-DOF
        # correction. Replace per-module indexing with per-wall indexing,
        # and use the wall centroid (mean of all module centroids in that
        # wall) as the rotation pivot for the wall.
        unique_walls = np.array(sorted(set(int(t) for t in taggers)),
                                dtype=np.int32)
        wall_to_idx = {int(w): i for i, w in enumerate(unique_walls)}
        # Build a (per-module → wall index) array, then translate the
        # per-row module indices into per-row wall indices.
        mod_to_wall = np.array([wall_to_idx[int(t)] for t in taggers],
                               dtype=np.int32)
        m1i = mod_to_wall[m1i_mod]
        m2i = mod_to_wall[m2i_mod]
        # Same translation for the FULL (train + val) sample, used at
        # eval time after the fit completes.
        m1i_full_loss = mod_to_wall[m1i_mod_full]
        m2i_full_loss = mod_to_wall[m2i_mod_full]
        # Wall centroid = mean of all module centroids belonging to that
        # wall (data-driven, like the per-module case).
        wall_centroids = np.zeros((len(unique_walls), 3))
        for i, w in enumerate(unique_walls):
            mask = (taggers == w)
            wall_centroids[i] = centroids[mask].mean(axis=0)
        n_modules = len(unique_walls)  # repurpose `n_modules` slot for walls
        centroids = wall_centroids
        # For the prior shift below: turn it into a per-wall init.
        wall_taggers = unique_walls
        is_wall_mode = True
        print(f"  WALL MODE: {len(unique_walls)} walls, "
              f"taggers = {unique_walls.tolist()}", flush=True)
    else:
        m1i = m1i_mod
        m2i = m2i_mod
        m1i_full_loss = m1i_mod_full
        m2i_full_loss = m2i_mod_full
        n_modules = len(mod_ids)
        wall_taggers = taggers
        is_wall_mode = False

    n_params = n_modules * 6 + 6

    loss_fn, unpack = make_loss(
        sp1, sp2, m1i, m2i, ap_nom, atpc,
        centroids, anode_centroids,
        n_modules, args.sigma_cm,
        args.prior_dT_cm, args.prior_dR_rad,
        fractional=args.fractional, frac_clamp_cm=args.frac_clamp_cm,
    )

    init_tags = set(int(t) for t in args.init_dz_tags.split(",") if t)
    init_theta = np.zeros(n_params, dtype=np.float64)
    if init_tags and args.init_dz_cm != 0.0:
        n_set = 0
        for i, tag in enumerate(wall_taggers):
            if int(tag) in init_tags:
                init_theta[i * 3 + 2] = args.init_dz_cm
                n_set += 1
        print(f"  initial guess: applied dz = {args.init_dz_cm:+.2f} cm to "
              f"{n_set} {'walls' if is_wall_mode else 'modules'} "
              f"in tagger(s) {sorted(init_tags)}", flush=True)

    pre_loss = float(loss_fn(jnp.zeros(n_params, dtype=jnp.float64)))
    print(f"  pre-alignment loss at theta=0 (incl reg): {pre_loss:.4f}", flush=True)

    bounds = None
    if args.no_rotations:
        # Translation slots: 0 .. 3*n_modules and 6*n_modules .. 6*n_modules+3.
        # Rotation slots: everything else. Pin rotation slots to 0.
        bounds = []
        for i in range(n_params):
            in_mod_T = (i < n_modules * 3)
            in_anode_T = (n_modules * 6 <= i < n_modules * 6 + 3)
            if in_mod_T or in_anode_T:
                bounds.append((None, None))
            else:
                bounds.append((0.0, 0.0))

    theta_opt, history, res = run_optimization(
        loss_fn, n_params, max_iter=args.max_iter, theta0=init_theta,
        bounds=bounds)
    print(f"  post-alignment loss: {res.fun:.4f} "
          f"(reduction × {pre_loss / max(res.fun, 1e-12):.2f})",
          flush=True)

    # Per-residual perpendicular distances pre and post (in cm) for plotting.
    def _rodrigues(dR_per_pt, v):
        ang2 = jnp.sum(dR_per_pt ** 2, axis=1, keepdims=True)
        small = ang2 < 1e-10
        safe_ang2 = jnp.where(small, 1.0, ang2)
        safe_ang = jnp.sqrt(safe_ang2)
        s = jnp.where(small, 1.0 - ang2 / 6.0, jnp.sin(safe_ang) / safe_ang)
        c = jnp.where(small, 0.5 - ang2 / 24.0, (1.0 - jnp.cos(safe_ang)) / safe_ang2)
        cross_tv = jnp.cross(dR_per_pt, v)
        return v + s * cross_tv + c * jnp.cross(dR_per_pt, cross_tv)

    def per_residual_distances(theta_np, sp1_, sp2_, m1i_, m2i_, ap_, atpc_):
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        mod_dT, mod_dR, anode_dT, anode_dR = unpack(theta)
        c1 = jnp.asarray(centroids)[m1i_]
        c2 = jnp.asarray(centroids)[m2i_]
        ca = jnp.asarray(anode_centroids)[atpc_]
        v1 = sp1_ - c1; P1 = _rodrigues(mod_dR[m1i_], v1) + c1 + mod_dT[m1i_]
        v2 = sp2_ - c2; P2 = _rodrigues(mod_dR[m2i_], v2) + c2 + mod_dT[m2i_]
        va = ap_ - ca; Pa = _rodrigues(anode_dR[atpc_], va) + ca + anode_dT[atpc_]
        line = P2 - P1
        u = line / (jnp.linalg.norm(line, axis=1, keepdims=True) + 1e-9)
        delta = Pa - P1
        proj = jnp.sum(delta * u, axis=1, keepdims=True) * u
        return np.asarray(jnp.linalg.norm(delta - proj, axis=1))

    # Train residuals (the ones the loss saw).
    res_pre_train  = per_residual_distances(np.zeros(n_params),
                                            sp1, sp2, m1i, m2i, ap_nom, atpc)
    res_post_train = per_residual_distances(theta_opt,
                                            sp1, sp2, m1i, m2i, ap_nom, atpc)
    # Validation residuals (held out — never seen by the loss).
    sp1_val = sp1_full[val_idx]; sp2_val = sp2_full[val_idx]
    m1i_val = m1i_full_loss[val_idx]; m2i_val = m2i_full_loss[val_idx]
    ap_val  = ap_nom_full[val_idx]; atpc_val = atpc_full[val_idx]
    res_pre_val  = per_residual_distances(np.zeros(n_params),
                                          sp1_val, sp2_val, m1i_val, m2i_val,
                                          ap_val, atpc_val)
    res_post_val = per_residual_distances(theta_opt,
                                          sp1_val, sp2_val, m1i_val, m2i_val,
                                          ap_val, atpc_val)
    # Legacy aliases (full sample, train + val concatenated in train-then-val
    # order) — kept so older plotting scripts that read `residuals_pre.npy`
    # and `residuals_post.npy` still work.
    res_pre = np.concatenate([res_pre_train, res_pre_val])
    res_post = np.concatenate([res_post_train, res_post_val])
    print(f"  median residual TRAIN: pre {np.median(res_pre_train):.3f} cm, "
          f"post {np.median(res_post_train):.3f} cm", flush=True)
    print(f"  median residual VAL  : pre {np.median(res_pre_val):.3f} cm, "
          f"post {np.median(res_post_val):.3f} cm", flush=True)
    np.save(os.path.join(args.outdir, "residuals_pre_train.npy"),  res_pre_train)
    np.save(os.path.join(args.outdir, "residuals_post_train.npy"), res_post_train)
    np.save(os.path.join(args.outdir, "residuals_pre_val.npy"),    res_pre_val)
    np.save(os.path.join(args.outdir, "residuals_post_val.npy"),   res_post_val)

    # Save per-residual arrays so the comparison plot can read them later.
    np.save(os.path.join(args.outdir, "residuals_pre.npy"), res_pre)
    np.save(os.path.join(args.outdir, "residuals_post.npy"), res_post)
    print(f"  saved residuals_pre.npy, residuals_post.npy "
          f"({len(res_pre):,} entries)", flush=True)

    if is_wall_mode:
        wall_counts = np.zeros(len(unique_walls), dtype=np.int64)
        for i, w in enumerate(unique_walls):
            wall_counts[i] = int(counts[taggers == w].sum())
        write_results(theta_opt, unique_walls, wall_counts, history,
                      args.outdir, kind="wall",
                      centroids=centroids, anode_centroids=anode_centroids)
        make_plots(np.zeros(n_params), theta_opt, res_pre, res_post,
                   unique_walls, wall_counts, history, args.outdir)
    else:
        write_results(theta_opt, mod_ids, counts, history, args.outdir,
                      centroids=centroids, anode_centroids=anode_centroids)
        make_plots(np.zeros(n_params), theta_opt, res_pre, res_post,
                   mod_ids, counts, history, args.outdir)
    print(f"  results written to {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
