# Handoff: building the double-anode-track displacement DataFrame

This document captures everything needed to extend the SBND CRT/drift
study with a **CRT-independent** field-distortion sample built from
"double anode-cathode crosser" cosmics — tracks that traverse the
entire detector from one anode to the other through the cathode.

The goal is a per-hit displacement DataFrame analogous to the existing
`crtdisp_df`, but with the track defined by its two anode-piercing
endpoints instead of two CRT space-points.

---

## 1. Project context

Working directory: `/exp/sbnd/app/users/gputnam/cathode-study/`

- Detector: **SBND** central-cathode geometry. Cathode at `x = 0`,
  anodes at `x = ±200 cm`. Active drift per TPC = 200 cm.
- Drift clock: 2 MHz → **500 ns/tick**. **Full drift = 2575 ticks**.
- Empirical anode tick peak: **418.5 ticks** (drift_t = 0 reference
  for hits inside the anode peak).
- E-field, T (sim defaults): 0.5 kV/cm, 88.4 K → `v_drift = 0.15627 cm/µs`.

### Constants (single source of truth: `crtlib/constants.py`)

```python
ANODE_X_NOMINAL  = {0: -200.0, 1: 200.0}
ANODE_CENTROID   = {0: (-200.0, 0.0, 250.0),
                    1: ( 200.0, 0.0, 250.0)}
FULL_DRIFT_TICKS = 2575.0
WHICHT0_NONE     = -1     # whicht0 enum: -1 = no t0 applied
SENTINEL         = -1e30  # CRT-track endpoint sentinel
```

### Anode-tick cut window (production)

`cut_tuning/cut_config_per_tpc.json`:

```json
{
  "anode_tick_peak":  418.5,
  "tpc_t_min_lo":     414.5,
  "tpc_t_min_hi":     427.5,
  "threshold_frac":   0.05
}
```

A track's earliest collection-plane hit time (`t_min`) in a TPC must
fall in `[414.5, 427.5]` ticks for that TPC to count as "anode-piercing".

### Input ntuples

Master filelist: **`filelist_nonempty_combined.txt`** — 137,445 paths
from the SBND `Gen2_intime_Run1` calibration production, both
`offbeamlight` (40,087) and `bnblight` (97,358) substreams.

(`filelist_nonempty.txt` is the smaller offbeamlight-only subset.
**Don't confuse the two** — use combined for all production analyses.)

xrootd auth is required for `/pnfs/...` paths. If you see
`Auth failed: No protocols left to try`, re-run `kx509` / `htgettoken`.

---

## 2. Double anode-cathode crosser selection

Per track (from raw `caloskim/TrackCaloSkim` ntuples; **no CRT
requirement**):

```python
DRIFT_FRAC = 0.95
drift_thresh = DRIFT_FRAC * FULL_DRIFT_TICKS   # = 2446.25 ticks

if whicht0 == WHICHT0_NONE:        continue   # need any t0 applied
if not (tpc0_n_p2_hits > 0
        and tpc1_n_p2_hits > 0):    continue   # hits in BOTH TPCs

# Per-TPC earliest / latest oncalo collection-plane hit time:
tpc0_t_min, tpc0_t_max = ...
tpc1_t_min, tpc1_t_max = ...

ac0 = (414.5 <= tpc0_t_min <= 427.5) and (tpc0_t_max - tpc0_t_min >= drift_thresh)
ac1 = (414.5 <= tpc1_t_min <= 427.5) and (tpc1_t_max - tpc1_t_min >= drift_thresh)

is_double_ac = ac0 and ac1   # ← the target selection
```

A double-AC track has:
- `t_min ≈ 418` (anode-piercing hit) in **both** TPCs
- `t_max − t_min ≈ 2575` (full drift span) in **both** TPCs
- Both endpoint coordinates: at `x = ±200` (the two anodes), with
  `(y, z)` from the earliest-tick collection-plane hit per TPC.

### Per-track endpoint extraction

For each TPC `i ∈ {0, 1}`:
1. From `hits2.h.time[i], hits2.h.tpc[i], hits2.oncalo[i]`,
   keep only `oncalo == True` collection-plane hits in TPC `i`.
2. Find the hit with minimum `time` → that's the **anode hit**.
3. Endpoint `Pi = (ANODE_X_NOMINAL[i], hits2.h.sp.y[j_min], hits2.h.sp.z[j_min])`.

This `(y, z)` is exactly the same anode-piercing point used in
`make_align_df_sbnd.py`'s `tpc{i}_anode_y, tpc{i}_anode_z` columns —
just don't restrict to either-TPC anode-piercing, require **both**.

---

## 3. Measured yields (probe on 400 ntuple files)

Script: `count_double_anode.py`. Runs in ~21 s.

```
  N tracks (whicht0 != -1)        : 85,409
  single anode-cathode crossers   : 1,062  (1.24  % of N tracks)
  double anode-cathode crossers   :    99  (0.116 % of N tracks)
```

Scaled to the full 137,445-file set (× 343.6):

| Selection | Tracks (full set) |
|---|---:|
| Single anode-cathode crosser | ~365,000 |
| **Double anode-cathode crosser** | **~34,000** |

For comparison:
- `align_df.parquet` (CRT-matched + anode-piercing, **either TPC**): **151,141 tracks**
- `crtdisp_df` (CRT-matched, **all hits** of those tracks): **4,784,856 unique tracks**, 1.07 B hits, ~225 hits/track

Double-AC is ~22 % of the align_df track count, ~0.7 % of crtdisp_df's
track count. The double-AC analysis is **not statistically competitive
for fine-voxel field-distortion maps** but is a strong systematic
cross-check (clean geometric model, both endpoints at known anode planes).

---

## 4. Building the double-anode displacement DataFrame

This is the analog of `crtdisp_df_part*.parquet` for the new sample.

### Schema (one row per hit, like crtdisp_df)

```
run, subrun, event, trk_id      uint32   event/track keys
hit_idx                         uint32   index within track
hit_x, hit_y, hit_z             float32  TPC space-point (oncalo, plane 2)
hit_tpc                         int8     0 or 1
hit_time                        float32  tick
anode0_y, anode0_z              float32  TPC0 anode-piercing point (y, z)
anode1_y, anode1_z              float32  TPC1 anode-piercing point (y, z)
foot_x, foot_y, foot_z          float32  foot of perpendicular on the
                                         anode-anode line (analogous
                                         to crt_x/y/z)
dx, dy, dz                      float32  hit_xyz - foot_xyz
whicht0                         uint8    t0 provenance
__file                          int32    source-file index (for grouping)
```

The anode-anode line is `P(t) = P0 + t · (P1 - P0)` with
`P0 = (-200, anode0_y, anode0_z)`, `P1 = (+200, anode1_y, anode1_z)`.

### Implementation plan

Adapt `make_crtdisp_df_sbnd.py` (the existing CRT-line displacement
builder) — most of the plumbing is reusable. Concretely, write
`make_doubleanode_df_sbnd.py` doing:

1. **Per-file loop** (multi-process, xrootd). Branches needed from
   `caloskim/TrackCaloSkim` (no crtana branches — `whicht0` is enough):

   ```python
   TCS_TRK_BRANCHES = ["meta.run", "meta.subrun", "meta.evt", "id",
                       "whicht0"]
   TCS_HIT_BRANCHES = ["hits2.h.time", "hits2.h.sp.x",
                       "hits2.h.sp.y", "hits2.h.sp.z",
                       "hits2.h.tpc", "hits2.oncalo"]
   ```

2. **Per-track selection (loop over tracks in the file):**

   - Skip if `whicht0 == -1`.
   - Per TPC `i ∈ {0, 1}`, compute `t_min, t_max, (anode_y, anode_z)` from
     `oncalo == True` collection-plane hits.
   - Require both TPCs `t_min` in `[414.5, 427.5]` AND
     `t_max - t_min ≥ 2446.25` (double-AC test).
   - Endpoints: `P0 = (-200, anode0_y, anode0_z)`,
                `P1 = (+200, anode1_y, anode1_z)`.

3. **Per-hit emission (for selected tracks):**

   For every `oncalo == True` collection-plane hit (both TPCs):

   ```python
   u = (P1 - P0) / np.linalg.norm(P1 - P0)         # unit direction
   delta = hit_xyz - P0
   foot  = P0 + (delta @ u) * u                   # foot of perpendicular
   d_vec = hit_xyz - foot                          # (dx, dy, dz)
   ```

   Append one row to the per-file DataFrame with all schema columns.

4. **Shard write** (ShardWriter from `make_crtdisp_df_sbnd.py` works
   as-is) — cap shards at ~100 M rows / ~4 GB.

5. **CLI** mirrors `make_crtdisp_df_sbnd.py`. **Default `inputs` to
   `filelist_nonempty_combined.txt`** (consistent with other
   builders since 2026-05-10).

### Expected output

- ~34k tracks × ~225 hits/track ≈ **~7.6 M total hits** (vs
  crtdisp_df's 1.07 B). Fits in **one shard** easily.
- Wallclock ≈ 1–2 h on `--nproc 48` (similar throughput to
  `make_align_df_sbnd.py` per file, but no crtana branches and
  cheaper per-track loop). Memory ≪ `make_crtdisp_df_sbnd.py` since
  the output is much smaller.

### Output path

Put it in the production location next to the existing parquets:
`/exp/sbnd/data/users/gputnam/efield-study/doubleanode_df/doubleanode_df_partNNN.parquet`

---

## 5. Downstream — voxel maps from the new DataFrame

`plot_crtdisp_voxels.py` reads `dx, dy, dz, hit_x, hit_y, hit_z` and
the CRT-subset boolean columns (`crt_has_S`, etc.) to make 4-D
histograms binned by `(x_slice, y, z, residual_bin)`. For the double-anode
DataFrame:

- **Drop the CRT-subset columns** (no taggers) — the script's SUBSETS
  machinery should be reduced or bypassed. A simple variant:
  `plot_doubleanode_voxels.py` that processes only ONE subset (all hits,
  no NS/EW slicing) since CRT-tagger geometry doesn't apply.
- Use the same `--voxel-size 20`, save `voxel_data.npz` (don't pass
  `--no-save-npz` — needed for downstream comparison plots).
- nproc note: on this host (125 GiB RAM, 64 CPU), peak memory is
  ~30 GiB regardless of `--nproc` for `plot_crtdisp_voxels.py` (the
  Pool is only used in the post-accumulation fit/plot stage). Use
  `--nproc 64`.

For a 6-pane-style comparison of the existing CRT-based and the new
double-AC voxel maps, copy `compare_ns_ew_6pane_v3.py` and point
VARIANTS at:

```python
VARIANTS = [
    ("CRT-matched, unaligned",       "crtdisp_maps_v3_20cm_unalign"),
    ("CRT-matched, module-aligned",  "crtdisp_maps_v3_20cm_module_gdml"),
    ("double-AC (CRT-free)",         "doubleanode_maps_20cm"),
]
```

(`compare_ns_ew_6pane.py` has a known Y/Z transpose bug — drop `.T`
on line ~101 if reusing.)

---

## 6. Key existing files / scripts to read first

| File | Why |
|---|---|
| `CLAUDE.md` | Project overview, file-system rules, environment, library map, cut history, nproc/memory notes for plot_crtdisp_voxels. |
| `crtlib/constants.py` | Single source of constants (anode positions, drift ticks, etc.) |
| `crtlib/ntuple.py` | Branch lists `TCS_TRK_BRANCHES_FULL`, `TCS_HIT_BRANCHES_FULL`. |
| `crtlib/io.py` | `pnfs_to_xroot`, `set_use_xroot`, `load_cut_config`, `collect_filelist`. |
| `count_double_anode.py` | Probe-yield script; mirrors per-track ntuple loop with the double-AC test. **Copy and extend this for the per-hit DataFrame builder.** |
| `make_align_df_sbnd.py` | Per-track ntuple → parquet builder. The per-TPC `t_min, t_max, anode_y, anode_z` extraction logic is exactly what's needed. |
| `make_crtdisp_df_sbnd.py` | Per-hit (CRT-projection) parquet builder + `ShardWriter`. Closest existing analog to the target script. |
| `plot_crtdisp_voxels.py` | Per-voxel `dx/dy/dz` histograms + Gaussian-fit μ/σ. Trivially adaptable. |

---

## 7. SBND geometry quick reference

```
Cathode plane         x =    0     (DO NOT use as a reference — non-flat)
TPC0 anode            x = -200 cm
TPC1 anode            x = +200 cm
Active y range        |y| < ~200 cm  (boxTPCActive y_full = 407 cm)
Active z range        0 < z < ~510 cm (boxTPCActive z_full = 509 cm)
TPC active half-width 200 cm  → drift time 2575 ticks ≈ 1287 µs
```

`tpc_{0,1}_anode_y, tpc_{0,1}_anode_z` in `align_df.parquet` is the
`(y, z)` of the **collection-plane (plane 2) hit at minimum tick** in
that TPC — the exact thing needed for the double-AC endpoints.

---

## 8. Filesystem rules (per CLAUDE.md)

**Never write outside this directory unless explicitly told.** Specifically:
- Do not modify `/exp/sbnd/data/...` unless rebuilding a parquet
  user has asked for.
- Do not modify `/exp/sbnd/app/users/.../sbndcode/` — read-only.
- Use `crtlib/` for shared code; don't duplicate constants.

---

## 9. State as of handoff (2026-05-10)

- `align_df.parquet` was rebuilt today from the full
  `filelist_nonempty_combined.txt` with the
  `threshold_frac = 0.05` cut window. **151,141 tracks.**
- Wall + module CRT-alignment fits (with 90/10 train/val split, fractional
  loss default-on, GDML pivots) have been run; results in
  `crt_alignment_v3_{wall,module}_gdml/`. B = 1000 bootstraps done.
- `crtdisp_df_part*.parquet` (4.78 M tracks, May 4) is the **older**
  CRT-projection-based per-hit DataFrame. The double-anode DataFrame
  to be built is its CRT-FREE complement, intended as a systematic
  cross-check on the existing field-distortion voxel maps.
