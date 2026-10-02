"""Per-module index, GDML-centroid Hungarian match, and JAX input packing.

These helpers operate on the residual-row DataFrame produced by
`crtlib.residuals.explode_to_residual_rows`.
"""

import numpy as np
import pandas as pd


def build_module_index(rows_df, df_full=None):
    """Return ``(mod_ids, mod_to_idx, centroids, counts, taggers)``:

      - mod_ids:    sorted array of unique module FEB MACs in `rows_df`
      - mod_to_idx: {mac -> index in [0, n_modules)}
      - centroids:  (n_modules, 3) mean of CRT space-point positions
                    per module (used as the rotation pivot for that
                    module's transform)
      - counts:     per-module residual count
      - taggers:    per-module dominant tagger ID, taken from
                    `df_full[crt_sp{1,2}_tagger]` if provided
    """
    macs = pd.concat([rows_df["mod1_id"], rows_df["mod2_id"]]).unique()
    mod_ids = np.array(sorted(macs.tolist()), dtype=np.int64)
    mod_to_idx = {int(m): i for i, m in enumerate(mod_ids)}
    n = len(mod_ids)

    sp1 = rows_df[["crt_sp1_x", "crt_sp1_y", "crt_sp1_z"]].to_numpy()
    sp2 = rows_df[["crt_sp2_x", "crt_sp2_y", "crt_sp2_z"]].to_numpy()
    m1 = rows_df["mod1_id"].to_numpy()
    m2 = rows_df["mod2_id"].to_numpy()

    sums = np.zeros((n, 3))
    counts = np.zeros(n, dtype=np.int64)
    for arr, mods in ((sp1, m1), (sp2, m2)):
        for j in range(len(arr)):
            i = mod_to_idx[int(mods[j])]
            sums[i] += arr[j]
            counts[i] += 1
    centroids = sums / np.maximum(counts, 1)[:, None]

    taggers = np.full(n, -1, dtype=np.int32)
    if df_full is not None:
        for col_mac, col_tag in [("crt_sp1_module_id", "crt_sp1_tagger"),
                                 ("crt_sp2_module_id", "crt_sp2_tagger")]:
            sub = df_full[[col_mac, col_tag]].dropna()
            for mac, tag in zip(sub[col_mac], sub[col_tag]):
                m = int(mac)
                if m in mod_to_idx and taggers[mod_to_idx[m]] < 0:
                    taggers[mod_to_idx[m]] = int(tag)
    return mod_ids, mod_to_idx, centroids, counts, taggers


def load_gdml_centroids(mod_ids, empirical_centroids,
                        gdml_csv="crt_module_gdml_centroids.csv"):
    """Match each empirical module (FEB MAC) to its nearest GDML module
    via Hungarian one-to-one assignment, return the GDML world-frame
    centroid array in the SAME ORDER as `mod_ids`.

    Used to replace data-driven (mean-of-spacepoints) module centroids
    with the geometry-defined ones, so rotation pivots correspond to
    the design module centers rather than cosmic-flux-biased means.
    """
    from scipy.optimize import linear_sum_assignment
    gdml = pd.read_csv(gdml_csv)
    gdml_xyz = gdml[["world_x", "world_y", "world_z"]].to_numpy()
    n_emp = len(empirical_centroids)
    n_gdml = len(gdml_xyz)
    if n_emp > n_gdml:
        raise ValueError(
            f"More empirical modules ({n_emp}) than GDML modules ({n_gdml}); "
            f"cannot do one-to-one match.")
    diff = empirical_centroids[:, None, :] - gdml_xyz[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    out = np.zeros_like(empirical_centroids)
    out[row_ind] = gdml_xyz[col_ind]
    md = float(np.median(np.linalg.norm(out - empirical_centroids, axis=1)))
    print(f"  GDML matching: {n_emp} empirical → {n_gdml} GDML modules, "
          f"median match dist = {md:.2f} cm", flush=True)
    return out


def pack_inputs_for_jax(rows_df, mod_to_idx):
    """Return jnp arrays consumed by the loss functions in
    `crtlib.jax_loss`. Order: (sp1, sp2, mod1_idx, mod2_idx,
    anode_p_nom, anode_tpc)."""
    import jax.numpy as jnp
    sp1 = rows_df[["crt_sp1_x", "crt_sp1_y", "crt_sp1_z"]].to_numpy(np.float64)
    sp2 = rows_df[["crt_sp2_x", "crt_sp2_y", "crt_sp2_z"]].to_numpy(np.float64)
    mod1_idx = np.array([mod_to_idx[int(m)]
                         for m in rows_df["mod1_id"].to_numpy()],
                        dtype=np.int32)
    mod2_idx = np.array([mod_to_idx[int(m)]
                         for m in rows_df["mod2_id"].to_numpy()],
                        dtype=np.int32)
    anode_x = rows_df["anode_x_nom"].to_numpy(np.float64)
    anode_y = rows_df["anode_y"].to_numpy(np.float64)
    anode_z = rows_df["anode_z"].to_numpy(np.float64)
    anode_p_nom = np.stack([anode_x, anode_y, anode_z], axis=1)
    anode_tpc = rows_df["anode_tpc"].to_numpy(np.int32)
    return (jnp.asarray(sp1), jnp.asarray(sp2),
            jnp.asarray(mod1_idx), jnp.asarray(mod2_idx),
            jnp.asarray(anode_p_nom), jnp.asarray(anode_tpc))
