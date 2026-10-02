#!/usr/bin/env python3
"""Score, post-process and RANK the dx-closure campaign variants.

Each campaign run (run_dx_campaign.sh) produces a voxel_data.npz and an
accumulator dump (M = Sigma n n^T, v = Sigma n m, plus the X-cut
accumulator when --min-abs-slope was active). For every run this script
scores, against the as-applied truth (`delta_applied` in the
build_apparent_truth_map.py npz):

  * bare      -- the run's own map as produced;
  * x_ridge*  -- dx re-solved with an anisotropic x-ridge, with the ridge
                 strength CALIBRATED to dx slope = 1 on the odd-z voxel
                 half and VALIDATED on the even-z half. Applied to the
                 3x3 solve (no-cut runs) or to the conditional X-subset
                 solve (runs with acc_x in the dump);
  * scale*    -- dx multiplied by k* = 1/slope(odd-z half), validated on
                 the even-z half.

Optionally each run carries an A-variant control map (same intervention on
the true-arrival-time sample): its dx slope measures how much REAL signal
the intervention destroys.

Outputs: campaign_metrics.csv (one row per variant), two ranked lists
(best MC closure; best data-applicable -- excluding MC-calibrated ridge /
scale / truth-derived LUTs), and a 1D dx-vs-x overlay of the top variants.

Usage:
    python rank_dx_interventions.py \
        --truth apparent_truth_map_mc_v2.npz \
        --run "R0:none::campaign/R0" \
        --run "R1:cut0.8:cut:campaign/R1" ... \
        --out campaign_results
Run spec: LABEL:DESCRIPTION:TAGS:DIR[:A_CONTROL_DIR]
  TAGS comma-set from {cut,detrend,lut,lut2} -- 'lut'/'lut2' mark
  MC-truth-derived corrections (not data-applicable).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B

COMPONENTS = ("dx", "dy", "dz")
RIDGE_GRID = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5,
                       0.7, 1.0])


def slope_rms(meas, truth, mask, sig_thresh):
    mm, tt = meas[mask], truth[mask]
    ok = np.isfinite(mm) & np.isfinite(tt)
    mm, tt = mm[ok], tt[ok]
    sig = np.abs(tt) > sig_thresh
    sl = (mm[sig] * tt[sig]).sum() / (tt[sig] ** 2).sum() \
        if sig.sum() > 5 else np.nan
    rms = np.sqrt(np.mean((mm - tt) ** 2)) if mm.size else np.nan
    return sl, rms, int(ok.sum())


def halves(shape):
    """odd-z (calibration) / even-z (validation) voxel masks."""
    iz = np.arange(shape[2])[None, None, :]
    odd = np.broadcast_to(iz % 2 == 1, shape)
    return odd, ~odd


def load_dump(path):
    d = np.load(path)
    out = dict(M6=d["nnT"].astype(np.float64),
               v=d["sum_nm"].astype(np.float64),
               n=d["n"].astype(np.float64))
    if "nnT_x" in d:
        out.update(M6x=d["nnT_x"].astype(np.float64),
                   vx=d["sum_nm_x"].astype(np.float64),
                   n_x=d["n_x"].astype(np.float64))
    return out


def solve_dx_ridge(dump, dy, dz, lam, min_n):
    """dx per voxel with x-ridge lam*n: conditional on (dy, dz) from the
    X-cut accumulator if present, else from the full 3x3 system."""
    if "M6x" in dump:
        T = dump["M6x"]; v = dump["vx"]; n = dump["n_x"]
        Mxx = T[..., 0] + lam * n
        ok = (n >= min_n) & (Mxx > 1e-6) & np.isfinite(dy)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ok, (v[..., 0] - T[..., 3] * dy
                                 - T[..., 4] * dz) / Mxx, np.nan)
    T = dump["M6"]; v4 = dump["v"]; n = dump["n"]
    sh = n.shape
    M = np.zeros((*sh, 3, 3))
    M[..., 0, 0] = T[..., 0] + lam * n
    M[..., 1, 1] = T[..., 1]; M[..., 2, 2] = T[..., 2]
    M[..., 0, 1] = M[..., 1, 0] = T[..., 3]
    M[..., 0, 2] = M[..., 2, 0] = T[..., 4]
    M[..., 1, 2] = M[..., 2, 1] = T[..., 5]
    ok = n >= min_n
    out = np.full(sh, np.nan)
    Mf = M[ok]; vf = v4[ok]
    try:
        sol = np.linalg.solve(Mf, vf[..., None])[..., 0]
        out[ok] = sol[:, 0]
    except np.linalg.LinAlgError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default="apparent_truth_map_mc_v2.npz")
    ap.add_argument("--run", action="append", default=[],
                    metavar="LABEL:DESC:TAGS:DIR[:ADIR]")
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--common-keep", action="store_true",
                    help="score every variant on the INTERSECTION of all "
                         "runs' valid-voxel sets (fair cross-variant "
                         "comparison; removes coverage-selection effects)")
    ap.add_argument("--out", default="campaign_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t = np.load(args.truth)
    truth = t["delta_applied"].astype(np.float64)
    sh = truth.shape[:3]
    odd, even = halves(sh)
    vx_c = B.voxel_centers(t["x_edges"])

    common = None
    if args.common_keep:
        for spec in args.run:
            rdir = spec.split(":")[3]
            d = np.load(os.path.join(rdir, "voxel_data.npz"))
            mu = d["delta_mu_all"].astype(np.float64)
            k = (np.isfinite(mu).all(-1) & np.isfinite(truth).all(-1)
                 & (d["count_all"] >= args.min_counts))
            common = k if common is None else (common & k)
        print(f"common voxel set: {common.sum()}", flush=True)

    rows = []
    profiles = {}

    for spec in args.run:
        parts = spec.split(":")
        label, desc, tags, rdir = parts[0], parts[1], parts[2], parts[3]
        adir = parts[4] if len(parts) > 4 else None
        tagset = set(filter(None, tags.split(",")))
        d = np.load(os.path.join(rdir, "voxel_data.npz"))
        mu = d["delta_mu_all"].astype(np.float64)
        cnt = d["count_all"]
        keep = (np.isfinite(mu).all(-1) & np.isfinite(truth).all(-1)
                & (cnt >= args.min_counts))
        if common is not None:
            keep = common
        dump_path = os.path.join(rdir, "accum.npz")
        dump = load_dump(dump_path) if os.path.exists(dump_path) else None

        # A-variant signal-retention control
        a_slope = np.nan
        if adir and os.path.exists(os.path.join(adir, "voxel_data.npz")):
            da = np.load(os.path.join(adir, "voxel_data.npz"))
            mua = da["delta_mu_all"].astype(np.float64)
            ka = (np.isfinite(mua).all(-1) & np.isfinite(truth).all(-1)
                  & (da["count_all"] >= args.min_counts))
            a_slope, _, _ = slope_rms(mua[..., 0], truth[..., 0], ka, 0.15)

        def record(vname, mu_use, data_ok, extra=""):
            r = dict(run=label, desc=desc, variant=vname,
                     tags=",".join(sorted(tagset)) or "-",
                     data_applicable=data_ok, a_slope=a_slope,
                     n_vox=int(keep.sum()), note=extra)
            for c, comp in enumerate(COMPONENTS):
                st = 0.15 if c == 0 else 0.3
                sl_v, rms_v, _ = slope_rms(mu_use[..., c], truth[..., c],
                                           keep & even, st)
                r[f"{comp}_slope"] = sl_v
                r[f"{comp}_rms"] = rms_v
            rows.append(r)
            profiles[f"{label}/{vname}"] = np.nanmean(
                np.where(keep[..., None], mu_use, np.nan)[..., 0],
                axis=(1, 2))

        # data applicability: cut/detrend are in-situ; LUTs are MC-derived
        base_data_ok = not (tagset & {"lut", "lut2"})
        record("bare", mu, base_data_ok)

        # ramp*: in-situ linear (v_AC vs v_DA) correction -- fitted from the
        # map itself with anode+cathode anchoring; DATA-APPLICABLE.
        try:
            import fit_dx_ramp as FR
            rrf = FR.fit_ramp_npz(os.path.join(rdir, "voxel_data.npz"),
                                  min_counts=args.min_counts)
            eps_f = np.nan_to_num(np.array(
                [rrf["fits"][0].get("eps", np.nan),
                 rrf["fits"][1].get("eps", np.nan)]))
            vxc = B.voxel_centers(np.load(
                os.path.join(rdir, "voxel_data.npz"))["x_edges"])
            mu_ramp = mu.copy()
            mu_ramp[..., 0] = mu[..., 0] - FR.ramp_grid(vxc, eps_f)[:, None,
                                                                   None]
            record("ramp*", mu_ramp, base_data_ok,
                   f"eps={eps_f[0]*100:+.2f}/{eps_f[1]*100:+.2f}cm/m")
        except Exception as e:
            print(f"  {label}: ramp* failed ({type(e).__name__}: {e})")

        if dump is not None:
            dy_ = mu[..., 1]; dz_ = mu[..., 2]
            # ridge calibration on the odd-z half
            sl_o = []
            for lam in RIDGE_GRID:
                dx_l = solve_dx_ridge(dump, dy_, dz_, lam, args.min_counts)
                s, _, _ = slope_rms(dx_l, truth[..., 0], keep & odd, 0.15)
                sl_o.append(s)
            sl_o = np.array(sl_o)
            ok_l = np.isfinite(sl_o)
            lam_star = np.nan
            if ok_l.sum() > 2 and (sl_o[ok_l].min() < 1.0 < sl_o[ok_l].max()):
                lam_star = float(np.interp(1.0, sl_o[ok_l][::-1],
                                           RIDGE_GRID[ok_l][::-1]))
            if np.isfinite(lam_star):
                mu_r = mu.copy()
                mu_r[..., 0] = solve_dx_ridge(dump, dy_, dz_, lam_star,
                                              args.min_counts)
                record("ridge*", mu_r, False, f"lam_x*={lam_star:.3f}")

        # scale calibration on the odd-z half
        s_odd, _, _ = slope_rms(mu[..., 0], truth[..., 0], keep & odd, 0.15)
        if np.isfinite(s_odd) and abs(s_odd) > 0.05:
            mu_s = mu.copy()
            mu_s[..., 0] = mu[..., 0] / s_odd
            record("scale*", mu_s, False, f"k*={1.0/s_odd:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "campaign_metrics.csv"), index=False)

    # ---- rankings -------------------------------------------------------------
    df["closure_score"] = (np.abs(df.dx_slope - 1.0)
                           + 0.5 * np.clip(df.dx_rms - 0.30, 0, None)
                           + 2.0 * np.clip(0.95 - df.dy_slope, 0, None)
                           + 2.0 * np.clip(0.95 - df.dz_slope, 0, None)
                           + 1.0 * np.clip(0.75 - df.a_slope.fillna(1.0),
                                           0, None))
    pd.set_option("display.width", 250)
    cols = ["run", "desc", "variant", "dx_slope", "dx_rms", "dy_slope",
            "dy_rms", "dz_slope", "dz_rms", "a_slope", "note",
            "closure_score"]
    r_all = df.sort_values("closure_score")
    r_data = df[df.data_applicable].sort_values("closure_score")
    print("\n===== RANKING 1: best MC closure (validated even-z half) =====")
    print(r_all[cols].to_string(index=False,
                                float_format=lambda x: f"{x:.3f}"))
    print("\n===== RANKING 2: best DATA-APPLICABLE (no MC-calibrated "
          "ridge/scale/LUT) =====")
    print(r_data[cols].to_string(index=False,
                                 float_format=lambda x: f"{x:.3f}"))
    r_all[cols].to_csv(os.path.join(args.out, "ranking_mc_closure.csv"),
                       index=False)
    r_data[cols].to_csv(os.path.join(args.out, "ranking_data_applicable.csv"),
                        index=False)

    # ---- 1D dx overlay of the top variants -----------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    tprof = np.nanmean(np.where(np.isfinite(truth[..., 0]),
                                truth[..., 0], np.nan), axis=(1, 2))
    ax.plot(vx_c, tprof, "r-", lw=2.5, label="as-applied truth")
    cmap = matplotlib.colormaps["tab10"]
    top = list(r_all.head(6).apply(
        lambda r: f"{r['run']}/{r['variant']}", axis=1))
    for i, key in enumerate(top):
        if key in profiles:
            ax.plot(vx_c, profiles[key], "o-", ms=3, lw=1.1,
                    color=cmap(i % 10), alpha=0.85, label=key)
    ax.axhline(0, color="0.6", ls=":", lw=0.7)
    ax.set_xlabel("X [cm]")
    ax.set_ylabel(r"$\Delta_x$ [cm]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("dx closure campaign: top variants vs the as-applied field")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "top_variants_dx_profile.png"),
                dpi=120)
    plt.close(fig)
    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
