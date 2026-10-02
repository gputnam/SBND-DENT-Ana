# Handoff: Rebuild `data/sbnd_cathode_v6.txt`

Drop this file (or its contents) into a new Claude chat to reproduce the SBND
v6 cathode volume export with a different input sample.

---

## Goal

Produce a single ASCII file `data/sbnd_cathode_v6.txt` that encodes the data-
driven SBND cathode volume as two surfaces `x_neg(y,z)` and `x_pos(y,z)` on a
fixed (y,z) grid. The C++ loader uses it to test whether `(x,y,z)` is inside
the cathode.

## Inputs

You'll be given **one of**:

1. An endpoints `.npz` like `endpoints_cathode_nocut.npz` with arrays
   `x, y, z` (cm, float32) and `whicht0` (int16). Each entry is one cathode-
   pair endpoint candidate from a reco-2/calibration ntuple.
2. A reco-2 ROOT file or calibration ntuple — in which case the first job is
   to extract those same four arrays and dump an `.npz`. Ask the user for the
   tree path and branch names if not obvious.

## v6 selection (DO NOT CHANGE without asking)

Apply both cuts to every endpoint before binning:

- `whicht0 >= 1`  (i.e. has a real T0 assignment, not the default)
- `|x| < 5 cm`    (drop endpoints clearly off the cathode plane)

Typical yield on the reference sample: ~759k endpoints out of ~1.6M raw.

## Grid

Hard-code these — the loader expects them:

```
NY     = 160
NZ     = 200
Y_MIN  = -200.0   Y_MAX = +200.0   (cm)
Z_MIN  =    0.0   Z_MAX = +500.0   (cm)
BIN_CM =    2.5
```

Bin centres: `y_c[j] = Y_MIN + (j+0.5)*BIN_CM`, same for z.

## Surface construction (v6 method — thickness-aware, no fill-holes)

For each (y-bin, z-bin):

1. Collect all endpoints with `(y,z)` falling in that 2.5x2.5 cm cell.
2. If the cell has fewer than `NMIN = 5` endpoints -> emit `nan` for both
   `x_neg` and `x_pos`.
3. Otherwise compute the **5th** and **95th** percentile of `x` in the cell:
   - `x_neg[j,k] = percentile(x_cell, 5)`
   - `x_pos[j,k] = percentile(x_cell, 95)`

   These two surfaces bracket the panel thickness; v6 keeps them separate
   instead of collapsing to a single mean (v5 behaviour).
4. **Do NOT** fill or interpolate NaN cells. The loader treats NaN as
   "outside the cathode", which is the intended behaviour.

## Output file format

Plain ASCII. Header lines, then two row-major grids in `(y, z)` order.

```
# SBND cathode v6 volume, exported from cathode_volume_v6.npz
# Units: centimetres.  Coordinates are (x,y,z) with x normal to cathode.
# A point (x,y,z) is *inside* the cathode volume when
#    Y_MIN <= y <= Y_MAX,  Z_MIN <= z <= Z_MAX,
#    x_neg(y,z) <= x <= x_pos(y,z),
# where x_neg and x_pos are read/interpolated from the grids below.
# A NaN value in either grid means "no data" at that (y,z) bin;
# the C++ loader treats NaN as "outside the volume".
VERSION  1
NY       160
NZ       200
Y_MIN    -200.000000
Y_MAX    200.000000
Z_MIN    0.000000
Z_MAX    500.000000
BIN_CM   2.500000
X_NEG
<160 rows, each with 200 space-separated floats (or "nan"), printf "%.4f">
X_POS
<160 rows, each with 200 space-separated floats (or "nan"), printf "%.4f">
```

Row index = y-bin (ascending y), column index = z-bin (ascending z). Print
`nan` (lower-case) for empty cells.

## Reference Python (drop-in)

```python
import numpy as np

NY, NZ = 160, 200
Y_MIN, Y_MAX = -200.0, 200.0
Z_MIN, Z_MAX =    0.0, 500.0
BIN          = 2.5
NMIN         = 5
PLO, PHI     = 5.0, 95.0

d  = np.load("endpoints_cathode_nocut.npz")
x, y, z, wt = d["x"], d["y"], d["z"], d["whicht0"]
m = (wt >= 1) & (np.abs(x) < 5.0)
x, y, z = x[m], y[m], z[m]

jy = np.floor((y - Y_MIN) / BIN).astype(np.int32)
kz = np.floor((z - Z_MIN) / BIN).astype(np.int32)
ok = (jy >= 0) & (jy < NY) & (kz >= 0) & (kz < NZ)
x, jy, kz = x[ok], jy[ok], kz[ok]

xneg = np.full((NY, NZ), np.nan)
xpos = np.full((NY, NZ), np.nan)
# group by bin
order = np.lexsort((kz, jy))
x, jy, kz = x[order], jy[order], kz[order]
# split indices where (jy,kz) changes
flat = jy.astype(np.int64) * NZ + kz
edges = np.flatnonzero(np.diff(flat)) + 1
starts = np.r_[0, edges]
ends   = np.r_[edges, len(flat)]
for s, e in zip(starts, ends):
    if e - s < NMIN: continue
    xb = x[s:e]
    j, k = jy[s], kz[s]
    xneg[j, k] = np.percentile(xb, PLO)
    xpos[j, k] = np.percentile(xb, PHI)

with open("data/sbnd_cathode_v6.txt", "w") as f:
    f.write("# SBND cathode v6 volume, exported from cathode_volume_v6.npz\n")
    f.write("# Units: centimetres.  Coordinates are (x,y,z) with x normal to cathode.\n")
    f.write("# A point (x,y,z) is *inside* the cathode volume when\n")
    f.write("#    Y_MIN <= y <= Y_MAX,  Z_MIN <= z <= Z_MAX,\n")
    f.write("#    x_neg(y,z) <= x <= x_pos(y,z),\n")
    f.write("# where x_neg and x_pos are read/interpolated from the grids below.\n")
    f.write("# A NaN value in either grid means \"no data\" at that (y,z) bin;\n")
    f.write("# the C++ loader treats NaN as \"outside the volume\".\n")
    f.write(f"VERSION  1\nNY       {NY}\nNZ       {NZ}\n")
    f.write(f"Y_MIN    {Y_MIN:.6f}\nY_MAX    {Y_MAX:.6f}\n")
    f.write(f"Z_MIN    {Z_MIN:.6f}\nZ_MAX    {Z_MAX:.6f}\n")
    f.write(f"BIN_CM   {BIN:.6f}\n")
    for tag, grid in (("X_NEG", xneg), ("X_POS", xpos)):
        f.write(tag + "\n")
        for row in grid:
            f.write(" ".join("nan" if np.isnan(v) else f"{v:.4f}" for v in row) + "\n")
```

## Sanity checks before declaring victory

- File has exactly `9 header lines` + `1 + NY` (X_NEG block) + `1 + NY`
  (X_POS block) data lines.
- Number of NaN cells in each grid is similar (panel coverage is symmetric).
- A handful of central bins (e.g. `y=0, z=250`) have `|x_neg|, |x_pos| < 4`.
- Make a quick imshow of `xpos - xneg` -- panel thickness should look ~0.5-3 cm
  across the active region, with sensible voids where the support bars sit.

## Things the new Claude should ASK before changing

- Different y/z range or bin size -> ask first; the C++ loader assumes the
  values above.
- Different percentile choice or `NMIN` -> ask first.
- Fill-hole / smoothing -> v5 did that; v6 deliberately does not. Ask.

## Deliverable location

Write the final file to `data/sbnd_cathode_v6.txt` (relative to the
sbndcode `CathodeFilter` package, i.e. it ends up installed via
`install_fw(LIST sbnd_cathode_v6.txt)`).
