"""SBND CRT/TPC constants and sentinels.

Single source of truth for values copied across scripts. Do not duplicate
these in script files — `from crtlib.constants import X` instead.
"""

import os

# ---------- xroot / file I/O ----------
XROOT_PREFIX = "root://fndca1.fnal.gov:1094/pnfs/fnal.gov/usr"

# CRT-track endpoints in crtana ntuples are sentinel-filled when unmatched.
# The float-min sentinel is around -1.8e308; check `< SENTINEL` to detect.
SENTINEL = -1e30

# whicht0 enum (sbncode TrackCaloSkimmer_module.cc):
#   -1 : no T0 source applied (raw hits)
#    0 : PFP T0 (Pandora flash match)
#    1 : CRT-Track T0
#    2 : CRT-Hit T0 (legacy)
#    3 : CRT-SpacePoint T0
WHICHT0_NONE = -1

# ---------- TPC geometry ----------
# Anode X positions per TPC. TPC0 anode at X=-200, TPC1 anode at X=+200,
# cathode at X=0 (central, NON-FLAT — DO NOT use as a reference plane).
ANODE_X_NOMINAL = {0: -200.0, 1: 200.0}
ANODE_CENTROID  = {0: (-200.0, 0.0, 250.0),
                   1: (200.0, 0.0, 250.0)}

# 2 MHz clock → 500 ns/tick. Active drift = 200 cm.
FULL_DRIFT_TICKS = 2575.0

# ---------- drift / timing (field + cathode pipelines) ----------
ANODE_TICK_T0   = 418.5      # empirical collection-plane anode peak tick
TICK_PERIOD_US  = 0.5        # 2 MHz TPC clock -> 500 ns = 0.5 us / tick
ACTIVE_DRIFT_CM = 200.0      # active drift length per TPC
# Drift velocity (cm/us). VDRIFT_MEASURED is the data-driven value the
# spatial/E-field/cathode pipelines use; VDRIFT_NOMINAL is the LArSoft sim
# default (use when reconstructing MC so x matches the simulation).
VDRIFT_MEASURED = 0.15532
VDRIFT_NOMINAL  = 0.15627

# ---------- external data locations (env-overridable) ----------
# Production parquets / intermediate products live under DATA_DIR at Fermilab.
# Point SBND_DATA_DIR elsewhere to relocate; every pipeline script also takes
# explicit --input / --outdir overrides.
DATA_DIR = os.environ.get(
    "SBND_DATA_DIR", "/exp/sbnd/data/users/gputnam/efield-study")

# Gen1 reference SCE ROOT (backward-displacement + E-field template copied
# into forward-only SCE files). Override with SBND_SCE_REFERENCE.
SCE_REFERENCE_ROOT = os.environ.get(
    "SBND_SCE_REFERENCE",
    "/cvmfs/sbnd.opensciencegrid.org/products/sbnd/sbnd_data/v01_03_00/"
    "SCEoffsets/SCEoffsets_SBND_E500_voxelTH3.root")

# ---------- CRT space-point construction ----------
# Tolerance for matching tpc_tr_start/end to a cl_sp position (cm).
SP_MATCH_TOL_CM = 0.5

# ---------- CRT taggers (walls) ----------
# Tagger ID → human-readable name. Source: sbndcode CRTCommonUtils.cc.
TAG_NAME = {
    0: "Bot", 1: "South", 2: "North", 3: "West",
    4: "East", 5: "TopLow", 6: "TopHigh",
}

# Wall normal axis index (0=x, 1=y, 2=z) per tagger ID. Verified
# empirically from align_df sample centroids.
#   Bot, TopLow, TopHigh    → ŷ
#   South, North            → ẑ
#   West, East              → x̂
WALL_NORMAL_AXIS = {0: 1, 1: 2, 2: 2, 3: 0, 4: 0, 5: 1, 6: 1}
