"""Per-wall residual comparison: pre / wall-GDML / module-GDML.

Reads four arrays per fit dir:
  residuals_pre_train.npy / residuals_post_train.npy   (90 % training sample)
  residuals_pre_val.npy   / residuals_post_val.npy     (10 % held-out)

For each sample (train, val) emits:
  - one combined linear-y histogram (1 cm bins) overlaying pre / wall / module
  - one per-wall histogram per CRT wall

Also produces a single fractional-improvement plot that overlays the
distribution of (d_post / d_pre) for the wall fit and the module fit
(both samples concatenated).

Outputs:
  residual_compare_per_wall/train/combined.{png,pdf}
  residual_compare_per_wall/train/wall_<id>_<name>.{png,pdf}
  residual_compare_per_wall/val/...
  residual_compare_per_wall/fractional_improvement.{png,pdf}
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crtlib.constants import TAG_NAME
from crtlib.io import load_align_df, load_cut_config
from crtlib.residuals import explode_to_residual_rows


def reproduce_row_filter(parquet, cut_config, max_pre_residual_cm, min_counts,
                          val_frac=0.1, val_seed=42):
    """Reapply the SAME cut chain + train/val split as align_crt.py.

    Returns ``(rows_train, rows_val)`` — both DataFrames with tag1, tag2
    columns. The order matches the order of residuals_*.npy arrays.
    """
    df = load_align_df(parquet)
    cuts = load_cut_config(cut_config)
    rows = explode_to_residual_rows(df, cuts["tmin_lo"], cuts["tmin_hi"],
                                    min_crt_len_cm=400.0)

    sp1 = rows[["crt_sp1_x", "crt_sp1_y", "crt_sp1_z"]].to_numpy()
    sp2 = rows[["crt_sp2_x", "crt_sp2_y", "crt_sp2_z"]].to_numpy()
    pa = np.stack([rows["anode_x_nom"].to_numpy(),
                   rows["anode_y"].to_numpy(),
                   rows["anode_z"].to_numpy()], axis=1)
    line = sp2 - sp1
    L = np.linalg.norm(line, axis=1, keepdims=True)
    u = line / np.maximum(L, 1e-9)
    delta = pa - sp1
    proj = np.sum(delta * u, axis=1, keepdims=True) * u
    pre_resid = np.linalg.norm(delta - proj, axis=1)
    rows = rows.loc[pre_resid <= max_pre_residual_cm].reset_index(drop=True)

    macs = pd.concat([rows["mod1_id"], rows["mod2_id"]]).astype(int)
    counts = macs.value_counts()
    keep_mods = set(counts[counts >= min_counts].index)
    keep_mask = (rows["mod1_id"].astype(int).isin(keep_mods) &
                 rows["mod2_id"].astype(int).isin(keep_mods))
    rows = rows[keep_mask].reset_index(drop=True)

    rng = np.random.default_rng(val_seed)
    is_val = rng.random(len(rows)) < val_frac
    rows_train = rows.loc[~is_val].reset_index(drop=True)
    rows_val   = rows.loc[is_val].reset_index(drop=True)
    return rows_train, rows_val


def overlay_hist(ax, pre, post_wall, post_mod, edges, labels, title):
    centers = 0.5 * (edges[:-1] + edges[1:])
    h_pre, _ = np.histogram(pre, bins=edges)
    h_w, _ = np.histogram(post_wall, bins=edges)
    h_m, _ = np.histogram(post_mod, bins=edges)
    ax.step(centers, h_pre, where="mid", lw=2.0, color="0.35", label=labels[0])
    ax.step(centers, h_w, where="mid", lw=1.7, color="tab:orange", label=labels[1])
    ax.step(centers, h_m, where="mid", lw=1.7, color="tab:blue", label=labels[2])
    ax.set_xlim(0, edges[-1]); ax.set_ylim(bottom=0)
    ax.set_xlabel("perpendicular residual (cm)")
    ax.set_ylabel("residual rows / bin")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)


def make_sample_plots(rows, pre, post_wall, post_mod, outdir,
                      sample_label, edges, n_label, max_per_wall_min=50,
                      per_wall_edges=None):
    os.makedirs(outdir, exist_ok=True)
    n_total = len(pre)
    pre_label  = f"pre-alignment (median {np.median(pre):.2f} cm)"
    wall_label = (f"wall fit, GDML (48 params) "
                  f"(median {np.median(post_wall):.2f} cm)")
    mod_label  = (f"module fit, GDML (528 params) "
                  f"(median {np.median(post_mod):.2f} cm)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    overlay_hist(ax, pre, post_wall, post_mod, edges,
                 (pre_label, wall_label, mod_label),
                 f"{sample_label} sample, all CRTs combined "
                 f"(N = {n_total:,})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"combined.{ext}"), dpi=140)
    plt.close(fig)

    summary = []
    tag1 = rows["tag1"].to_numpy(); tag2 = rows["tag2"].to_numpy()
    walls_sorted = sorted(set(int(t) for t in np.concatenate([tag1, tag2])
                              if t >= 0))
    pw_edges = per_wall_edges if per_wall_edges is not None else edges
    for w in walls_sorted:
        mask = (tag1 == w) | (tag2 == w)
        n = int(mask.sum())
        if n < max_per_wall_min:
            print(f"  [{sample_label}] wall {w} ({TAG_NAME.get(w, '?')}): "
                  f"only {n} entries — skipping plot")
            continue
        m_pre = float(np.median(pre[mask]))
        m_w   = float(np.median(post_wall[mask]))
        m_m   = float(np.median(post_mod[mask]))
        summary.append((w, TAG_NAME.get(w, str(w)), n, m_pre, m_w, m_m))
        pre_l = f"pre (median {m_pre:.2f} cm)"
        w_l   = f"wall fit (median {m_w:.2f} cm)"
        m_l   = f"module fit (median {m_m:.2f} cm)"
        fig, ax = plt.subplots(figsize=(9, 5.5))
        overlay_hist(ax, pre[mask], post_wall[mask], post_mod[mask], pw_edges,
                     (pre_l, w_l, m_l),
                     f"{sample_label} — Wall {w} ({TAG_NAME.get(w, '?')}) — "
                     f"N = {n:,}")
        fig.tight_layout()
        tagname = TAG_NAME.get(w, str(w))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"wall_{w}_{tagname}.{ext}"),
                        dpi=140)
        plt.close(fig)

    pd.DataFrame(summary, columns=["tagger", "name", "N",
                                   "median_pre", "median_wall", "median_mod"]
                 ).to_csv(os.path.join(outdir, "median_per_wall.csv"),
                          index=False)
    return summary


def fractional_improvement_plot(pre_full, post_wall_full, post_mod_full,
                                 outdir, n_bins=30, eps_cm=0.5):
    """Overlay distribution of (d_post / d_pre) for wall and module fits.

    Bin width is chosen wide enough to suppress statistical fluctuations:
    ``n_bins`` over [0, 2] (so default 30 → 0.067-wide bins).
    """
    safe_pre = np.maximum(pre_full, eps_cm)
    r_wall = post_wall_full / safe_pre
    r_mod  = post_mod_full  / safe_pre
    edges = np.linspace(0.0, 2.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h_w, _ = np.histogram(np.clip(r_wall, edges[0], edges[-1]), bins=edges)
    h_m, _ = np.histogram(np.clip(r_mod,  edges[0], edges[-1]), bins=edges)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.step(centers, h_w, where="mid", lw=2.0, color="tab:orange",
            label=f"wall fit  (median {np.median(r_wall):.2f}, "
                  f"frac improved {np.mean(r_wall < 1):.2f})")
    ax.step(centers, h_m, where="mid", lw=2.0, color="tab:blue",
            label=f"module fit (median {np.median(r_mod):.2f}, "
                  f"frac improved {np.mean(r_mod < 1):.2f})")
    ax.axvline(1.0, color="0.35", lw=1.0, linestyle="--",
               label="no change (post = pre)")
    ax.set_xlim(0.0, 2.0); ax.set_ylim(bottom=0)
    ax.set_xlabel(r"fractional displacement $d_\mathrm{post} / d_\mathrm{pre}$"
                  f"  (clamped: $d_\\mathrm{{pre}} \\geq {eps_cm:g}$ cm)")
    ax.set_ylabel("residual rows / bin")
    ax.set_title(f"Fractional improvement, N = {len(pre_full):,} residuals")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fractional_improvement.{ext}"),
                    dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--cut-config",
                    default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--wall-dir",   default="crt_alignment_v3_wall_gdml")
    ap.add_argument("--module-dir", default="crt_alignment_v3_module_gdml")
    ap.add_argument("--max-pre-residual-cm", type=float, default=50.0)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--xmax", type=float, default=50.0)
    ap.add_argument("--bin-cm", type=float, default=1.0,
                    help="bin width (cm) for the combined histogram")
    ap.add_argument("--per-wall-bin-cm", type=float, default=2.0,
                    help="bin width (cm) for the per-wall histograms; "
                         "default 2x bin-cm so per-wall plots are smoother")
    ap.add_argument("--frac-bins", type=int, default=30,
                    help="number of bins on [0, 2] for fractional plot")
    ap.add_argument("--frac-eps-cm", type=float, default=0.5,
                    help="floor on d_pre when building the d_post/d_pre ratio")
    ap.add_argument("--outdir", default="residual_compare_per_wall")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows_train, rows_val = reproduce_row_filter(
        args.parquet, args.cut_config,
        args.max_pre_residual_cm, args.min_counts,
        val_frac=args.val_frac, val_seed=args.val_seed)
    print(f"  rows: {len(rows_train):,} train, {len(rows_val):,} val")

    edges = np.arange(0.0, args.xmax + args.bin_cm, args.bin_cm)
    pw_edges = np.arange(0.0, args.xmax + args.per_wall_bin_cm,
                         args.per_wall_bin_cm)

    pres = {}
    for sample in ("train", "val"):
        pres[sample] = {}
        for tag, d in (("wall", args.wall_dir), ("module", args.module_dir)):
            pre_p  = os.path.join(d, f"residuals_pre_{sample}.npy")
            post_p = os.path.join(d, f"residuals_post_{sample}.npy")
            if not (os.path.exists(pre_p) and os.path.exists(post_p)):
                raise FileNotFoundError(
                    f"missing {pre_p} or {post_p} — has the new align_crt.py "
                    f"with --val-frac been run? Older fits only wrote "
                    f"residuals_pre.npy / residuals_post.npy.")
            pres[sample][tag] = (np.load(pre_p), np.load(post_p))

    # Both fits must have used the same train/val split — pre arrays must
    # match between dirs.
    for sample in ("train", "val"):
        pre_w = pres[sample]["wall"][0]; pre_m = pres[sample]["module"][0]
        if pre_w.shape != pre_m.shape:
            raise RuntimeError(f"{sample}: pre shapes differ "
                               f"({pre_w.shape} vs {pre_m.shape})")
        md = float(np.max(np.abs(pre_w - pre_m)))
        print(f"  {sample}: pre array consistency between dirs: max |Δ| = {md:.2e} cm")

    # Per-sample plots: absolute (pre + wall + module) and fractional
    # (d_post/d_pre overlay of wall vs module).
    summary_rows = []
    for sample, rows in (("train", rows_train), ("val", rows_val)):
        pre, post_wall = pres[sample]["wall"]
        _,   post_mod  = pres[sample]["module"]
        if len(pre) != len(rows):
            raise RuntimeError(
                f"{sample}: row count mismatch — reproduced filter {len(rows)} "
                f"vs residuals_pre {len(pre)}. Cut config or val-seed drift?")
        outdir = os.path.join(args.outdir, sample)
        sample_summary = make_sample_plots(
            rows, pre, post_wall, post_mod,
            outdir, sample.upper(), edges, n_label=sample,
            per_wall_edges=pw_edges)
        for entry in sample_summary:
            summary_rows.append((sample, *entry))
        # Fractional-improvement plot per sample.
        fractional_improvement_plot(
            pre, post_wall, post_mod, outdir,
            n_bins=args.frac_bins, eps_cm=args.frac_eps_cm)

    print(f"\nper-wall medians (cm):  (residual sliced by tag1==w OR tag2==w)")
    print(f"  {'sample':<6s} {'wall':<14s} {'N':>7s} {'pre':>8s} "
          f"{'wall':>8s} {'module':>8s}")
    for s, w, name, n, mp, mw, mm in summary_rows:
        print(f"  {s:<6s} {name:<14s} {n:>7,d} {mp:>8.3f} "
              f"{mw:>8.3f} {mm:>8.3f}")

    print(f"\n  wrote {args.outdir}/{{train,val}}/combined.{{png,pdf}}, "
          f"{args.outdir}/{{train,val}}/wall_*.{{png,pdf}}, "
          f"{args.outdir}/{{train,val}}/fractional_improvement.{{png,pdf}}")


if __name__ == "__main__":
    main()
