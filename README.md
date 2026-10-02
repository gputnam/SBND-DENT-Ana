# SBND field-distortion & cathode-distortion analyses

Two linked SBND detector-distortion measurements, built from raw wire/tick hit
positions of cosmic-muon tracks (no Pandora space points, no CRT):

1. **Field distortion** (2-D double-anode method) — a per-voxel 3-D spatial
   distortion map from muons that pierce **both** anodes, fit with a
   tensor-product cubic B-spline, converted to an **E-field** distortion layer,
   and assembled into a SBND `SpaceCharge` ROOT (forward + numerically-inverted
   backward + E-field).
2. **Cathode distortion** — the drift-coordinate displacement of the cathode
   plane from anode-cathode through-going tracks, plus the drift velocity, fit
   to a per-TPC 2-D spline and written as the SBND `CathodeFilter` ASCII map.

Both analyses share a small library, `crtlib`, and a common set of SBND
constants and cuts. The flat top-level `*.py` files are the pipeline stages;
exploratory / deprecated work lives under [`experiments/`](experiments/).

---

## Quick start

```bash
# 1. Build the environment (any machine with Python >=3.9 + pip/PyPI access)
./bootstrap_venv.sh
source venv/bin/activate          # or: source setup.sh  (on a Fermilab gpvm)

# 2. Run the test suite (synthetic unit tests + a real-data mini-run)
pytest -q                          # unit tests only
pytest -q -m integration           # the end-to-end mini-run on committed fixtures
pytest -q -m ""                    # everything
```

On a **Fermilab SBND gpvm**, `source setup.sh` additionally loads the spack
HDF5 + XRootD needed to read remote `/pnfs` ntuples over xrootd. On any other
machine use `bootstrap_venv.sh`; remote-ntuple reads then need
`pip install fsspec-xrootd xrootd` (not required for the local mini-run or for
re-running any stage from existing parquets).

---

## Repository layout

```
crtlib/                 shared library
  constants.py            SBND geometry / timing / drift constants; env-overridable
                          data locations (DATA_DIR, SCE_REFERENCE_ROOT)
  io.py                   parquet / filelist / cut-config / wire-geometry loaders
  ntuple.py               caloskim branch lists + per-cluster helpers
  voxel_invert.py         per-voxel scalar (rank-1) + rank-2 LSQ accumulators
  spline.py               tensor-product cubic B-spline fit / eval / invert
  plotting.py             LIBIMAGEQUANT GIF helpers + per-X-slice 3-pane figures
config/                 wire_geometry_sbnd.json, cut_tuning/*.json (committed)
<field + cathode pipeline scripts>   flat at the repo top (run as python X.py)
drivers/                canonical pipeline shell drivers + dump_wire_geom.fcl
tests/                  pytest suite + committed real-data fixtures
docs/history/           per-epoch analysis reports + CHANGELOG
experiments/            ancillary / deprecated code (CRT alignment, dx forensics,
                        generators, diagnostics) — not covered by tests
```

Pipeline scripts stay flat at the top so they import each other directly and run
as `python <script>.py`. `pip install -e .` (done by `bootstrap_venv.sh`)
installs only the `crtlib` package.

---

## Data locations (intermediate & input files)

Large inputs and intermediate products are **not** committed; they live under a
single base directory, overridable with the `SBND_DATA_DIR` environment variable
(default is the Fermilab production area). Every script also takes explicit
`--input` / `--outdir` flags.

| Env var | Default | Meaning |
|---|---|---|
| `SBND_DATA_DIR` | `/exp/sbnd/data/users/gputnam/efield-study` | production parquets + intermediate maps/splines/ROOTs |
| `SBND_SCE_REFERENCE` | cvmfs `SCEoffsets_SBND_E500_voxelTH3.root` | Gen1 reference SCE ROOT (backward + E-field template) |

Key products under `$SBND_DATA_DIR` (see `docs/history/CHANGELOG.md` for the
version history):

| Path (under `$SBND_DATA_DIR`) | Contents |
|---|---|
| `doubleanode_2d_df*/` | per-hit parquets (raw plane/tpc/wire/time/integral) — the field-pipeline input |
| `doubleanode_2d_maps_*/` | spatial voxel maps (`voxel_data.npz`) + heatmaps/GIFs |
| `doubleanode_2d_spline_*/` | fitted spatial spline + per-X-slice plots |
| `efield_2d_{maps,spline,gauss}_*/` | E-field layer (voxel → spline → Gauss check) |
| `sce_input_doubleanode_2d/` | SCE ROOT deliverables (forward / E-field / combined) |
| `drift_df/` | per-track collection-plane start/end times — cathode + v_drift input |
| `cathode_study_2cm_v3/` | per-pixel CB cathode-tick fits (`cathode_fit_results_tpc{0,1}.csv`) |
| `cathode_pipeline/step{1,2,3}*/` | cathode displacement → spline → v6 ASCII |

Current physics deliverables (point the SBND services at these):
`sbnd_spacecharge.InputFilename` → the latest
`sce_input_doubleanode_2d/sbnd_sce_doubleanode_2d_*.root`; the cathode loader →
the latest `cathode_pipeline/step3_*/sbnd_cathode_v6_drift.txt`.

### Filelists

The large production filelists (30–60 MB of `/pnfs` paths) are **not** committed
(`example_filelist.txt` holds a 15-line sample). Regenerate / point at your own
with `crtlib.io.collect_filelist` or
`experiments/generators/build_filelist_v11.py`.

---

## Field-distortion pipeline

All stages read/write `.npz` / `.root`; the only remote-data step is stage 0
(building the per-hit parquet from ntuples). Representative commands — adjust
`--input`/`--outdir` and knot counts to taste (production values are in
`docs/history/CHANGELOG.md` and the `drivers/` scripts).

```bash
# 0. Build the per-hit parquet from caloskim ntuples (needs xrootd for /pnfs).
python make_doubleanode_2d_df_sbnd.py $SBND_DATA_DIR/doubleanode_2d_df \
       example_filelist.txt --nproc 48

# 1. (once) tune the anode-disagreement cut -> config/cut_tuning/anode_cut.json
python optimize_anode_cut.py --input $SBND_DATA_DIR/doubleanode_2d_df

# 2. Spatial voxel map (two-iteration outlier rejection + joint x-ramp fix).
python plot_doubleanode_2d_voxels.py \
       --input $SBND_DATA_DIR/doubleanode_2d_df --outdir maps_20cm \
       --voxel-size 20 --iterations 2 --iter-resid-cm 4.0 \
       --anode-extrapolate --x-ramp-correct hits --x-ramp-joint --nproc auto

# 3. Fit the spatial spline and write the forward SCE ROOT.
python fit_doubleanode_2d_spline.py --input maps_20cm/voxel_data.npz \
       --outdir spline_20cm --nx-knots 5 --ny-knots 6 --nz-knots 9 \
       --ridge-lambda 1 --write-sce-root --sce-root-name sbnd_sce_fwd.root

# 4. Derive the E-field layer from the spatial spline and write its SCE ROOT.
python build_efield_2d_voxels.py --input maps_20cm/voxel_data.npz \
       --spline-knots-json spline_20cm/spline_knots.json --outdir ef_maps_20cm
python fit_efield_2d_spline.py --input ef_maps_20cm/voxel_data.npz \
       --outdir ef_spline_20cm --nx-knots 8 --ny-knots 10 --nz-knots 16 \
       --ridge-lambda 0.1 --write-sce-root \
       --displacement-root <sce-dir>/sbnd_sce_fwd.root \
       --sce-root-name sbnd_sce_efield.root

# 5. Assemble the combined SCE ROOT (forward + inverted backward + E-field).
python build_sce_root.py --forward-root <sce-dir>/sbnd_sce_fwd.root \
       --efield-root <sce-dir>/sbnd_sce_efield.root \
       --out <sce-dir>/sbnd_sce_doubleanode_2d.root

# optional: Gauss's-law consistency check on the E-field map
python check_gauss_2d.py --input ef_maps_20cm/voxel_data.npz --outdir ef_gauss_20cm
```

MC closure (compare the measured map to the simulated truth):
`compare_2d_vs_truth.py` and `plot_1d_profiles_vs_truth.py`.

The full production chain is wrapped in `drivers/run_pipeline_data_v10c.sh` /
`drivers/run_pipeline_mc_full_v10jc.sh`.

## Cathode-distortion pipeline

```bash
# 0. Build the per-track drift DataFrame (needs xrootd for /pnfs).
python make_driftV_df_sbnd.py drift_df.parquet example_filelist.txt --nproc 48

# 1. Per-(y,z) cathode-tick map + CB drift-velocity fit.
python plot_cathode_heatmap.py --input drift_df.parquet --outdir cathode_study_2cm
python fit_driftv_cb.py --input drift_df.parquet          # drift velocity (CB fit)

# 2. tick -> cm displacement, per TPC.
python cathode_step1_tick_to_cm.py --csv-dir cathode_study_2cm --outdir step1

# 3. Per-TPC 2-D spline.
python cathode_step2_spline.py --step1-dir step1 --outdir step2 \
       --ny-knots 80 --nz-knots 120 --ridge-lambda 5

# 4. Assemble the v6 ASCII map for the SBND CathodeFilter loader.
python cathode_step3_v6.py --step1-dir step1 --step2-dir step2 --outdir step3
```

Driver: `drivers/run_cathode_pipeline_v11.sh`.

---

## Environment notes

- **Python** ≥ 3.9. Core dependencies: numpy, scipy, pandas, pyarrow, uproot,
  awkward, matplotlib, pillow, tqdm, dill (see `requirements.txt`).
- **Remote ntuple reads** (`/pnfs` over xrootd) additionally need
  `fsspec-xrootd` + `xrootd` and a valid token/proxy (`htgettoken` / `kx509`).
  "Auth failed: No protocols left to try" means re-authenticate.
- **SBND constants** (single source of truth in `crtlib.constants`): 500 ns/tick
  (2 MHz), 200 cm active drift, anode peak tick 418.5, nominal E-field
  0.5 kV/cm; drift velocity `VDRIFT_MEASURED = 0.15532 cm/µs` (data) /
  `VDRIFT_NOMINAL = 0.15627 cm/µs` (sim — use when reconstructing MC).

---

## Tests

`pytest` covers every stage of both analyses:

- **Unit** (fast, synthetic): voxel inversion, spline fit/eval/derivative + the
  build_sce_input / cathode_step2 delegation to `crtlib.spline`, the
  drift-velocity model, wire geometry, the gradient→E-field inversion, the
  forward→backward displacement inversion, SCE ROOT write/read, the cathode
  tick→cm conversion, the dx-ramp estimator, and the anode-extrapolation fix.
- **Integration** (`-m integration`): a miniature real-data run of each full
  pipeline on the committed fixtures in `tests/fixtures/` (a 1 500-track
  downsample + the Gen1 reference TH3F + a CB-fit CSV band). Regenerate the
  fixtures on a gpvm with `python tests/fixtures/_build_fixtures.py`.
