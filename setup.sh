#!/bin/bash
# Fermilab / SBND gpvm environment. Source this file:
#     source setup.sh
#
# Loads the spack HDF5 + XRootD that uproot needs to read remote /pnfs
# ntuples over xrootd, then activates the local Python venv. On a machine
# WITHOUT cvmfs/spack (laptop, generic cluster) use bootstrap_venv.sh
# instead and skip the spack loads below.
#
# After sourcing you can run any pipeline script directly, e.g.
#     python plot_doubleanode_2d_voxels.py --help

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ -d /cvmfs/larsoft.opensciencegrid.org ]; then
    source /cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.1.0/spack_env/setup-env.sh
    spack load /cszcdti  # hdf5@1.12.2  %gcc@12.5.0 arch=linux-almalinux9-x86_64_v2
    spack load /mqueum7  # xrootd@5.8.4 %gcc@12.5.0 arch=linux-almalinux9-x86_64_v2
else
    echo "setup.sh: /cvmfs not found -- skipping spack (reading remote /pnfs" \
         "over xrootd will be unavailable; local parquet/ROOT still work)." >&2
fi

if [ -d "$_here/venv" ]; then
    source "$_here/venv/bin/activate"
else
    echo "setup.sh: no venv/ yet -- run ./bootstrap_venv.sh first." >&2
fi
