# CLAUDE.md — SBND field / cathode distortion repo

Guidance for Claude Code (and humans) working in this repository. See
`README.md` for end-to-end run instructions and `docs/history/CHANGELOG.md` for
the full method history (v4–v11, MC closure, dx-closure campaign).

## What this repo is

Two linked SBND detector-distortion analyses built from raw wire/tick hit
positions (no Pandora space points, no CRT):

1. **Field distortion (2-D double-anode)** — spatial voxel map → cubic B-spline
   → E-field layer → combined SCE ROOT (forward + numerically-inverted backward
   + E-field).
2. **Cathode distortion** — cathode-tick map → per-TPC displacement spline → v6
   ASCII; plus the drift velocity from a Crystal-Ball fit of anode-cathode tick
   lengths.

The flat top-level `*.py` files are the pipeline stages; `crtlib/` is the shared
library; `experiments/` holds deprecated / exploratory code (CRT alignment, dx
forensics, generators, diagnostics) and is **not** maintained or tested.

## File-system rules (REQUIRED)

- **Never write outside this repository unless the user explicitly asks in the
  current conversation.** In particular do not modify files under
  `$SBND_DATA_DIR` (`/exp/sbnd/data/...`), `sbndcode/` source, or any home dir.
  Read-only access to those paths is fine.
- Production parquets / maps / ROOTs under `$SBND_DATA_DIR` are read-only unless
  the user explicitly asks for a rebuild. Scan large parquets in shards.
- Regenerate test fixtures only with `tests/fixtures/_build_fixtures.py`.

## Environment

`source setup.sh` (Fermilab gpvm) loads spack HDF5 + XRootD and activates
`venv/`. On any other machine use `./bootstrap_venv.sh` (plain venv +
`pip install -r requirements.txt -e .`). Remote `/pnfs` reads over xrootd need
`fsspec-xrootd` + `xrootd` and a valid token (`htgettoken`/`kx509`); "Auth
failed: No protocols left to try" = re-authenticate.

Data locations are env-overridable (`crtlib.constants`): `SBND_DATA_DIR`
(default `/exp/sbnd/data/users/gputnam/efield-study`) and `SBND_SCE_REFERENCE`
(the Gen1 reference SCE TH3F). Every script also takes `--input`/`--outdir`.

## `crtlib/` — shared library

| Module | Contents |
|---|---|
| `constants` | SBND geometry/timing/drift constants, env-overridable data paths, sentinels |
| `io` | `pnfs_to_xroot`, `collect_filelist`, `load_cut_config`, `load_wire_geometry`, `load_align_df` |
| `ntuple` | caloskim branch lists + `dominant_feb_mac`/`match_cluster_index` |
| `voxel_invert` | per-voxel scalar (rank-1, **active**) + rank-2 LSQ accumulators |
| `spline` | tensor-product cubic B-spline: `clamped_knots`, `univariate_basis`, `khatri_rao_sparse`, `fit_tensor_spline`, `eval_spline_grid`, `eval_spline_deriv`, `invert_displacement_grid` |
| `plotting` | `quantize_rgb`/`make_gif` (LIBIMAGEQUANT), `plot_xslice_3pane` |

`build_sce_input` and `cathode_step2_spline` both delegate their spline math to
`crtlib.spline` (previously byte-identical copies). `build_sce_input` re-exports
the shared primitives under their historical names, so
`import build_sce_input as B; B.univariate_basis(...)` still works.

## SBND constants (single source of truth: `crtlib.constants`)

| Constant | Value |
|---|---|
| `TICK_PERIOD_US` | 0.5 (2 MHz clock → 500 ns/tick) |
| `ACTIVE_DRIFT_CM` | 200 |
| `ANODE_TICK_T0` | 418.5 (empirical collection anode peak) |
| nominal E-field | 0.5 kV/cm |
| `VDRIFT_MEASURED` | 0.15532 cm/µs (data; spatial/E-field/cathode pipelines) |
| `VDRIFT_NOMINAL` | 0.15627 cm/µs (LArSoft sim — use for MC reco) |

Current cuts live in `config/`: anode-piercing window `cut_config_per_tpc.json`
(peak 418.5, window [414.5, 427.5]); anode-disagreement cut `anode_cut.json`
(0.891 cm). Wire geometry: `config/wire_geometry_sbnd.json` (pitch 0.300 cm;
collection at 0°, induction at ±60°, induction views swap between TPCs).

## Pipelines

See `README.md` for exact commands. In short:

- **Field:** `make_doubleanode_2d_df_sbnd.py` → `plot_doubleanode_2d_voxels.py`
  (`--iterations 2 --anode-extrapolate --x-ramp-correct hits --x-ramp-joint`) →
  `fit_doubleanode_2d_spline.py --write-sce-root` → `build_efield_2d_voxels.py` →
  `fit_efield_2d_spline.py --write-sce-root` → `build_sce_root.py`. Production
  spatial knots 5×6×9 (λ=1); E-field knots 8×10×16 (λ=0.1).
- **Cathode:** `make_driftV_df_sbnd.py` → `plot_cathode_heatmap.py` /
  `fit_driftv_cb.py` → `cathode_step1_tick_to_cm.py` → `cathode_step2_spline.py`
  (80×120 knots, λ=5) → `cathode_step3_v6.py`.

Drivers for the full chains are in `drivers/`.

## Conventions worth preserving

- **SCE X sign flip:** `SpaceChargeServiceSBND` flips `TrueFwd_Displacement_X`
  for x<0; the spline/voxel SCE writers store `-Δx` for TPC0 / `+Δx` for TPC1.
  Backward is stored in the **same** per-TPC flip as forward (verified by
  forward∘backward round-trip closure ~0.005 cm).
- **GIFs:** use `crtlib.plotting.quantize_rgb` (LIBIMAGEQUANT + Floyd-Steinberg);
  the default median-cut collapses light pastel heatmap shades.
- **Multiprocessing:** scripts that create multiple Pools in sequence with live
  matplotlib/numpy in the parent must `multiprocessing.set_start_method("spawn",
  force=True)` — a forked Pool inherits held locks and deadlocks on teardown.

## Tests

`pytest -q` (unit) and `pytest -q -m integration` (real-data mini-run on
`tests/fixtures/`). Add a per-stage test when adding a stage; keep the
`build_sce_input`/`cathode_step2` → `crtlib.spline` delegation intact (there is a
regression test for it in `tests/test_spline.py`).
