#!/usr/bin/env python3
"""In-situ estimator of the double-anode dx linear-ramp artifact.

The double-anode dx artifact is a per-TPC linear ramp in drift distance,
dx_fake = eps * d (equivalent to an effective drift-velocity offset:
v_DA = v_used / (1 + eps)). It is identifiable WITHOUT truth because the
physical Delta_x vanishes at BOTH ends of the drift: at the anode (zero
drift time) and at the cathode (enforced by the anode-cathode v_drift
calibration, v_AC = 2L/<dt>, which by construction puts full-drift charge
at the cathode). A function vanishing at both endpoints has no
anode-to-cathode linear component, so the ramp coefficient of the measured
map is pure artifact and the physical dome passes through untouched.

Per TPC, this module fits the quality-gated, y/z-averaged dx profile vs
drift distance d with

    dx(d) = eps * s_t * d  +  sum_k c_k B_k(d)

where s_t = +1 (TPC0) / -1 (TPC1) maps the drift-frame ramp into the
detector frame, and B_k(d) is a clamped cubic-spline "dome" basis ZEROED at
d = 0 and d = L (both end columns dropped) -- the ramp is the only term
nonzero at the cathode. A one-parameter near-cathode secant fit is reported
as a stability cross-check. Region splits (low/high z, low/high y) test the
uniformity of the ramp.

Usage (offline, on any voxel_data.npz):
    python fit_dx_ramp.py --input <maps>/voxel_data.npz --out <dir> \
        [--region-splits]

Importable: fit_ramp_from_arrays(...) and fit_ramp_npz(...) are used by
plot_doubleanode_2d_voxels.py --x-ramp-correct.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B

L_DRIFT = 200.0
ANODE_X = (-200.0, 200.0)


def dome_basis(d, n_int=4):
    """Clamped cubic-spline basis on d in [0, L] with BOTH end columns
    dropped (every retained element vanishes at d=0 and d=L)."""
    kx = B.clamped_knots(0.0, L_DRIFT, n_int)
    A = B.univariate_basis(np.asarray(d, float), kx)
    return A[:, 1:-1]


def cathode_basis(d, n_int=4):
    """The single clamped-cubic boundary column that is 1 at the cathode
    (d=L) and 0 at the anode (d=0) -- the basis the dome drops. Used as the
    SHARED mean-cathode-offset term: it lets the cathode value float while
    keeping the anode pegged exactly to 0 (returns shape (N, 1))."""
    kx = B.clamped_knots(0.0, L_DRIFT, n_int)
    A = B.univariate_basis(np.asarray(d, float), kx)
    return A[:, -1:]


def profile_dx(mu, cov, cond, cnt, vx, *, cond_max=100.0, max_res=2.0,
               min_counts=30, region_mask=None):
    """Weighted y/z-average of the dx layer per x column.

    Returns (prof, err, nvox) arrays over the x columns; NaN where empty.
    Weights 1/sigma_x^2 with the standard quality gates."""
    sigx2 = np.maximum(cov[..., 0], 1e-4)
    res = np.sqrt(np.maximum(cov[..., :3], 0.0)).max(axis=-1)
    good = (np.isfinite(mu[..., 0]) & np.isfinite(cond) & (cond <= cond_max)
            & np.isfinite(res) & (res <= max_res) & (cnt >= min_counts))
    if region_mask is not None:
        good = good & region_mask
    w = np.where(good, 1.0 / sigx2, 0.0)
    sw = w.sum(axis=(1, 2))
    swx = (w * np.nan_to_num(mu[..., 0])).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(sw > 0, swx / sw, np.nan)
        err = np.where(sw > 0, 1.0 / np.sqrt(sw), np.nan)
    return prof, err, good.sum(axis=(1, 2))


def fit_ramp_profile(vx, prof, err, *, fit_xmin=0.0, fit_xmax=190.0,
                     n_int=4, joint=False, cathode_offset=False):
    """Ramp+dome fit on a dx(x) profile.

    Per TPC the model is  dx(d) = eps * s_t * d + dome_t(d), with the dome
    basis zeroed at both drift ends. With joint=False each TPC's eps is fit
    independently. With joint=True a SINGLE shared eps is fit across both
    TPCs (separate dome shapes), so the correction is the TPC-AVERAGED ramp:
    the detector-common artifact survives the average while a genuine
    per-TPC drift-velocity difference (opposite-sign ramp in the two TPCs)
    cancels and is left in the map.

    cathode_offset=True (joint only) adds ONE shared mean-cathode-offset
    term c*E(d) (E = the cathode boundary basis, 1 at d=L, 0 at d=0). This
    frees the TPC-MEAN (symmetric) cathode value -- exactly the channel a
    genuine per-TPC drift-velocity difference lands in -- while the anode
    stays pegged at 0 and the antisymmetric cathode value stays solely eps
    (so eps remains identifiable). Per-TPC effective ramps are then
    eps +/- c/L; the implied v split is reported by the CLI.

    Returns dict per tpc: eps, eps_err, chi2, ndof, plus the fitted
    component curves for plotting."""
    if joint:
        return _fit_ramp_joint(vx, prof, err, fit_xmin=fit_xmin,
                               fit_xmax=fit_xmax, n_int=n_int,
                               cathode_offset=cathode_offset)
    out = {}
    for t, (xa, s_t) in enumerate(((ANODE_X[0], 1.0), (ANODE_X[1], -1.0))):
        inh = (vx < 0) if t == 0 else (vx > 0)
        d = np.abs(vx - xa)
        m = (inh & np.isfinite(prof) & np.isfinite(err)
             & (np.abs(vx) >= fit_xmin) & (np.abs(vx) <= fit_xmax))
        if m.sum() < n_int + 3:
            out[t] = dict(eps=np.nan, eps_err=np.nan, chi2=np.nan, ndof=0)
            continue
        Adome = dome_basis(d[m], n_int)
        A = np.column_stack([s_t * d[m], Adome])
        w = 1.0 / err[m]
        Aw = A * w[:, None]
        yw = prof[m] * w
        ATA = Aw.T @ Aw
        ATy = Aw.T @ yw
        try:
            covp = np.linalg.inv(ATA)
        except np.linalg.LinAlgError:
            out[t] = dict(eps=np.nan, eps_err=np.nan, chi2=np.nan, ndof=0)
            continue
        p = covp @ ATy
        resid = (prof[m] - A @ p) * w
        chi2 = float((resid ** 2).sum())
        ndof = int(m.sum() - len(p))
        scale = np.sqrt(max(chi2 / max(ndof, 1), 1.0))
        # secant cross-check: 1-par fit on the cathode-adjacent columns
        msec = (inh & np.isfinite(prof) & np.isfinite(err)
                & (np.abs(vx) >= fit_xmin) & (np.abs(vx) <= 60.0))
        eps_sec = np.nan
        if msec.sum() >= 2:
            ws = 1.0 / err[msec] ** 2
            eps_sec = float((ws * prof[msec] * s_t * d[msec]).sum()
                            / (ws * (d[msec]) ** 2).sum())
        out[t] = dict(eps=float(p[0]), eps_err=float(np.sqrt(covp[0, 0])
                                                     * scale),
                      eps_secant=eps_sec, chi2=chi2, ndof=ndof,
                      coeffs=p, mask=m, d=d, s_t=s_t, n_int=n_int)
    return out


def _fit_ramp_joint(vx, prof, err, *, fit_xmin, fit_xmax, n_int,
                    cathode_offset=False):
    """Shared-eps joint fit across both TPCs (TPC-averaged ramp).

    With cathode_offset=True an extra SHARED column c*E(d) (E = cathode
    boundary basis) frees the TPC-mean cathode value; column layout is then
    [eps | c | dome0(nb) | dome1(nb)] instead of [eps | dome0 | dome1]."""
    nb = dome_basis(np.array([50.0]), n_int).shape[1]
    noff = 1 if cathode_offset else 0
    base = 1 + noff                      # # of shared leading columns
    ncol = base + 2 * nb
    blocks, rows_A, rows_y, rows_w = [], [], [], []
    for t, (xa, s_t) in enumerate(((ANODE_X[0], 1.0), (ANODE_X[1], -1.0))):
        inh = (vx < 0) if t == 0 else (vx > 0)
        d = np.abs(vx - xa)
        m = (inh & np.isfinite(prof) & np.isfinite(err)
             & (np.abs(vx) >= fit_xmin) & (np.abs(vx) <= fit_xmax))
        blocks.append((t, s_t, d, m))
        if m.sum() == 0:
            continue
        Adome = dome_basis(d[m], n_int)
        blk = np.zeros((int(m.sum()), ncol))
        blk[:, 0] = s_t * d[m]
        if cathode_offset:
            blk[:, 1] = cathode_basis(d[m], n_int)[:, 0]   # shared, same sign
        blk[:, base + t * nb:base + (t + 1) * nb] = Adome
        rows_A.append(blk)
        rows_y.append(prof[m])
        rows_w.append(1.0 / err[m])
    if not rows_A:
        return {t: dict(eps=np.nan, eps_err=np.nan, chi2=np.nan, ndof=0)
                for t in (0, 1)}
    A = np.vstack(rows_A)
    y = np.concatenate(rows_y)
    w = np.concatenate(rows_w)
    Aw = A * w[:, None]
    try:
        covp = np.linalg.inv(Aw.T @ Aw)
    except np.linalg.LinAlgError:
        return {t: dict(eps=np.nan, eps_err=np.nan, chi2=np.nan, ndof=0)
                for t in (0, 1)}
    p = covp @ (Aw.T @ (y * w))
    resid = (y - A @ p) * w
    chi2 = float((resid ** 2).sum())
    ndof = int(len(y) - len(p))
    scale = np.sqrt(max(chi2 / max(ndof, 1), 1.0))
    eps = float(p[0])
    eps_err = float(np.sqrt(covp[0, 0]) * scale)
    coff = float(p[1]) if cathode_offset else 0.0
    coff_err = float(np.sqrt(covp[1, 1]) * scale) if cathode_offset else 0.0
    out = {}
    for t, s_t, d, m in blocks:
        ct = np.concatenate([[eps], p[base + t * nb:base + (t + 1) * nb]])
        out[t] = dict(eps=eps, eps_err=eps_err, eps_secant=np.nan,
                      cathode_offset=coff, cathode_offset_err=coff_err,
                      chi2=chi2, ndof=ndof, coeffs=ct, mask=m, d=d,
                      s_t=s_t, n_int=n_int)
    return out


def fit_ramp_npz(npz_path_or_dict, *, region_mask=None, fit_xmin=0.0,
                 fit_xmax=190.0, n_int=4, cond_max=100.0, max_res=2.0,
                 min_counts=30, joint=False, cathode_offset=False):
    """Convenience: load a voxel_data.npz (path or open dict) -> per-TPC
    ramp fit results + the profile used."""
    d = (np.load(npz_path_or_dict) if isinstance(npz_path_or_dict, str)
         else npz_path_or_dict)
    mu = d["delta_mu_all"].astype(np.float64)
    cov = d["delta_cov_all"].astype(np.float64)
    cond = d["cond_all"]
    cnt = d["count_all"]
    vx = B.voxel_centers(d["x_edges"])
    vy = B.voxel_centers(d["y_edges"])
    vz = B.voxel_centers(d["z_edges"])
    prof, err, nv = profile_dx(mu, cov, cond, cnt, vx, cond_max=cond_max,
                               max_res=max_res, min_counts=min_counts,
                               region_mask=region_mask)
    fits = fit_ramp_profile(vx, prof, err, fit_xmin=fit_xmin,
                            fit_xmax=fit_xmax, n_int=n_int, joint=joint,
                            cathode_offset=cathode_offset)
    vdrift = float(d["vdrift"]) if "vdrift" in d else np.nan
    return dict(fits=fits, prof=prof, err=err, vx=vx, vy=vy, vz=vz,
                vdrift=vdrift)


def ramp_grid(vx, eps01):
    """The fitted ramp evaluated on the x columns (detector frame):
    eps*(x - x_anode) per TPC half. eps01 = (eps_tpc0, eps_tpc1)."""
    out = np.where(vx < 0, eps01[0] * (vx - ANODE_X[0]),
                   -eps01[1] * (ANODE_X[1] - vx))
    return out


def region_masks(vy, vz):
    """The uniformity splits: half-splits in y and z plus the four
    (y, z) quadrants (broadcastable 3-D masks)."""
    Y = vy[None, :, None]
    Z = vz[None, None, :]
    sh = (1, len(vy), len(vz))
    lo_y, hi_y = Y < 0.0, Y >= 0.0
    lo_z, hi_z = Z < 250.0, Z >= 250.0
    return {
        "z<250": np.broadcast_to(lo_z, sh),
        "z>=250": np.broadcast_to(hi_z, sh),
        "y<0": np.broadcast_to(lo_y, sh),
        "y>=0": np.broadcast_to(hi_y, sh),
        "y<0,z<250": np.broadcast_to(lo_y & lo_z, sh),
        "y<0,z>=250": np.broadcast_to(lo_y & hi_z, sh),
        "y>=0,z<250": np.broadcast_to(hi_y & lo_z, sh),
        "y>=0,z>=250": np.broadcast_to(hi_y & hi_z, sh),
    }


def plot_fit(res, title, path):
    vx, prof, err, fits = res["vx"], res["prof"], res["err"], res["fits"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for t, ax in enumerate(axes):
        f = fits[t]
        inh = (vx < 0) if t == 0 else (vx > 0)
        ax.errorbar(vx[inh], prof[inh], yerr=err[inh], fmt="o", ms=4,
                    color="navy", label="dx profile (quality-gated)")
        if np.isfinite(f.get("eps", np.nan)) and "coeffs" in f:
            m = f["mask"]
            xx = vx[m]
            ramp = f["eps"] * f["s_t"] * f["d"][m]
            Adome = dome_basis(f["d"][m], f["n_int"])
            dome = Adome @ f["coeffs"][1:]
            coff = f.get("cathode_offset", 0.0)
            cterm = (coff * cathode_basis(f["d"][m], f["n_int"])[:, 0]
                     if coff else np.zeros_like(ramp))
            order = np.argsort(xx)
            ax.plot(xx[order], (ramp + dome + cterm)[order], "r-", lw=2,
                    label="fit: ramp + dome" + (" + offset" if coff else ""))
            ax.plot(xx[order], ramp[order], "--", color="tab:orange",
                    lw=1.6, label=f"ramp: eps={f['eps']*100:+.3f} "
                                  f"± {f['eps_err']*100:.3f} cm/m")
            ax.plot(xx[order], dome[order], ":", color="tab:green", lw=1.6,
                    label="dome (0 at anode & cathode)")
            if coff:
                ax.plot(xx[order], cterm[order], "-.", color="tab:purple",
                        lw=1.6, label=f"mean cathode offset c="
                                      f"{coff:+.3f} ± "
                                      f"{f.get('cathode_offset_err', 0.0):.3f} cm")
            ax.set_title(f"TPC{t}: secant x-check "
                         f"{f.get('eps_secant', np.nan)*100:+.3f} cm/m, "
                         f"chi2/ndof={f['chi2']:.1f}/{f['ndof']}",
                         fontsize=10)
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_xlabel("X [cm]")
        ax.set_ylabel(r"$\Delta_x$ [cm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="voxel_data.npz")
    ap.add_argument("--out", default=None,
                    help="output dir (default: alongside the npz)")
    ap.add_argument("--region-splits", action="store_true")
    ap.add_argument("--fit-xmin", type=float, default=0.0)
    ap.add_argument("--fit-xmax", type=float, default=190.0)
    ap.add_argument("--dome-knots", type=int, default=4)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--joint", action="store_true",
                    help="fit a single shared eps across both TPCs "
                         "(TPC-averaged ramp; preserves genuine per-TPC "
                         "drift-velocity differences in the map)")
    ap.add_argument("--cathode-offset", action="store_true",
                    help="add a shared mean-cathode-offset term c*E(d) to the "
                         "joint fit (frees the TPC-mean cathode value -> the "
                         "per-TPC drift-velocity-difference channel; anode "
                         "stays pegged, eps stays the antisymmetric ramp). "
                         "Implies --joint.")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.cathode_offset:
        args.joint = True
    out = args.out or os.path.dirname(args.input)
    os.makedirs(out, exist_ok=True)

    res = fit_ramp_npz(args.input, fit_xmin=args.fit_xmin,
                       fit_xmax=args.fit_xmax, n_int=args.dome_knots,
                       min_counts=args.min_counts, joint=args.joint,
                       cathode_offset=args.cathode_offset)
    v = res["vdrift"]
    summary = {"input": args.input, "vdrift_used": v, "regions": {}}
    print(f"{args.label or args.input}")
    print(f"{'region':10s} {'TPC':4s} {'eps [cm/m]':>14s} {'secant':>9s} "
          f"{'v_DA [cm/us]':>13s} {'chi2/ndof':>10s}")

    def report(tag, r):
        for t in (0, 1):
            f = r["fits"][t]
            eps = f.get("eps", np.nan)
            vda = v / (1.0 + eps) if np.isfinite(eps) and v == v else np.nan
            print(f"{tag:10s} TPC{t:1d} {eps*100:>+9.3f} ± "
                  f"{f.get('eps_err', np.nan)*100:.3f} "
                  f"{f.get('eps_secant', np.nan)*100:>+9.3f} {vda:>13.5f} "
                  f"{f.get('chi2', np.nan):>6.1f}/{f.get('ndof', 0)}",
                  flush=True)
        summary["regions"][tag] = {
            f"tpc{t}": {k: (float(r["fits"][t][k])
                            if np.isfinite(np.atleast_1d(
                                r["fits"][t].get(k, np.nan)).astype(float)).all()
                            else None)
                        for k in ("eps", "eps_err", "eps_secant")}
            for t in (0, 1)}

    report("all", res)
    if args.cathode_offset and np.isfinite(v):
        f0 = res["fits"][0]
        c = f0.get("cathode_offset", np.nan)
        c_err = f0.get("cathode_offset_err", np.nan)
        eps = f0.get("eps", np.nan)
        # per-TPC effective ramp eps +/- c/L -> per-TPC v_DA
        bL = c / L_DRIFT
        v0 = v / (1.0 + eps + bL)
        v1 = v / (1.0 + eps - bL)
        print(f"\nmean cathode offset c = {c:+.4f} ± {c_err:.4f} cm "
              f"(per-TPC velocity-difference channel)")
        print(f"  implied per-TPC effective ramp: TPC0 {((eps+bL)*100):+.3f}, "
              f"TPC1 {((eps-bL)*100):+.3f} cm/m")
        print(f"  implied v_DA: TPC0 {v0:.5f}  TPC1 {v1:.5f} cm/us "
              f"(split {v0-v1:+.5f})", flush=True)
        summary["cathode_offset_cm"] = float(c)
        summary["cathode_offset_err_cm"] = float(c_err)
        summary["v_da_split"] = {"tpc0": float(v0), "tpc1": float(v1)}
    plot_fit(res, f"dx ramp fit: {args.label or args.input}",
             os.path.join(out, "ramp_fit.png"))

    if args.region_splits:
        d0 = np.load(args.input)
        masks = region_masks(res["vy"], res["vz"])
        eps_split = {}
        for tag, msk in masks.items():
            rr = fit_ramp_npz(d0, region_mask=msk, fit_xmin=args.fit_xmin,
                              fit_xmax=args.fit_xmax, n_int=args.dome_knots,
                              min_counts=args.min_counts, joint=args.joint,
                              cathode_offset=args.cathode_offset)
            report(tag, rr)
            eps_split[tag] = rr
            safe = tag.replace("<", "lt").replace(">=", "ge").replace(",",
                                                                      "_")
            plot_fit(rr, f"dx ramp fit [{tag}]: {args.label or args.input}",
                     os.path.join(out, f"ramp_fit_{safe}.png"))
        # uniformity figure
        fig, ax = plt.subplots(figsize=(9, 5.5))
        tags = ["all"] + list(masks.keys())
        allres = {"all": res, **eps_split}
        xpos = np.arange(len(tags))
        for t, (color, off) in enumerate((("tab:blue", -0.12),
                                          ("tab:red", +0.12))):
            vals = [allres[g]["fits"][t].get("eps", np.nan) * 100
                    for g in tags]
            errs = [allres[g]["fits"][t].get("eps_err", np.nan) * 100
                    for g in tags]
            ax.errorbar(xpos + off, vals, yerr=errs, fmt="o", ms=6,
                        color=color, label=f"TPC{t}", capsize=3)
        ax.set_xticks(xpos)
        ax.set_xticklabels(tags)
        ax.set_ylabel(r"ramp $\epsilon$ [cm/m]")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"ramp uniformity: {args.label or args.input}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "ramp_uniformity.png"), dpi=120)
        plt.close(fig)

    with open(os.path.join(out, "ramp_fit.json"), "w") as jf:
        json.dump(summary, jf, indent=1)
    print(f"wrote {out}/ramp_fit.png{', ramp_uniformity.png' if args.region_splits else ''}, ramp_fit.json", flush=True)


if __name__ == "__main__":
    main()
