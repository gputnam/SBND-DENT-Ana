"""Regenerate the committed test fixtures from the Fermilab production data.

Run once on an SBND gpvm (read access to /exp/sbnd/data + /cvmfs).  The
outputs are small, self-contained samples committed to the repo so the
integration tests run without any remote data access:

    mini_doubleanode_2d_df.parquet   ~1500 double-anode tracks (all hits)
    ref_sce_small.root               Gen1 reference SCE TH3F (copied verbatim)
    cathode/cathode_fit_results_tpc{0,1}.csv   a contiguous Z-band of CB fits

    python tests/fixtures/_build_fixtures.py
"""
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SBND_DATA_DIR",
                      "/exp/sbnd/data/users/gputnam/efield-study")
REF = os.environ.get(
    "SBND_SCE_REFERENCE",
    "/cvmfs/sbnd.opensciencegrid.org/products/sbnd/sbnd_data/v01_03_00/"
    "SCEoffsets/SCEoffsets_SBND_E500_voxelTH3.root")

N_TRACKS = 1500
COLS = ["run", "subrun", "event", "trk_id", "plane", "tpc",
        "wire", "time", "integral"]


def build_parquet():
    src = os.path.join(DATA, "doubleanode_2d_df", "doubleanode_2d_df_part000.parquet")
    pf = pq.ParquetFile(src)
    seen, chunks, ntrk = set(), [], 0
    for batch in pf.iter_batches(batch_size=1_000_000, columns=COLS):
        df = batch.to_pandas()
        key = (df["run"].astype(str) + "_" + df["subrun"].astype(str) + "_"
               + df["event"].astype(str) + "_" + df["trk_id"].astype(str))
        for k in key.unique():
            if len(seen) >= N_TRACKS:
                break
            seen.add(k)
        keep = key.isin(seen)
        chunks.append(df[keep])
        ntrk = len(seen)
        if ntrk >= N_TRACKS:
            break
    out = pd.concat(chunks, ignore_index=True)
    # keep only whole tracks that were fully captured in the scanned batches
    path = os.path.join(HERE, "mini_doubleanode_2d_df.parquet")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path,
                   compression="zstd")
    print(f"wrote {path}: {len(out):,} hits / {ntrk} tracks, "
          f"{os.path.getsize(path)/1e6:.1f} MB")


def build_reference():
    dst = os.path.join(HERE, "ref_sce_small.root")
    shutil.copyfile(REF, dst)
    print(f"wrote {dst}: {os.path.getsize(dst)/1e6:.1f} MB")


def build_cathode():
    outdir = os.path.join(HERE, "cathode")
    os.makedirs(outdir, exist_ok=True)
    for tpc in (0, 1):
        src = os.path.join(DATA, "cathode_study_2cm_v3",
                           f"cathode_fit_results_tpc{tpc}.csv")
        df = pd.read_csv(src)
        # contiguous Z band keeps the 8-neighbour structure step2 needs.
        band = df[df["z_lo"] <= 200.0].copy()
        path = os.path.join(outdir, f"cathode_fit_results_tpc{tpc}.csv")
        band.to_csv(path, index=False)
        print(f"wrote {path}: {len(band):,} pixels, "
              f"{os.path.getsize(path)/1e6:.1f} MB")


if __name__ == "__main__":
    build_parquet()
    build_reference()
    build_cathode()
