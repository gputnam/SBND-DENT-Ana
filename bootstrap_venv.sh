#!/bin/bash
# Create the local Python virtual environment from scratch and install the
# package (editable) + its dependencies. Works on any machine with a
# Python >=3.9 and pip/PyPI access -- no cvmfs/spack required.
#
#     ./bootstrap_venv.sh            # uses `python3`
#     PYTHON=python3.11 ./bootstrap_venv.sh
#
# Afterwards activate with:  source venv/bin/activate   (or: source setup.sh)
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$_here"
PYTHON="${PYTHON:-python3}"

echo "[bootstrap] creating venv with $PYTHON ..."
"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .          # installs the crtlib package (editable)

echo "[bootstrap] done. Run the tests with:  pytest -q"
echo "[bootstrap] optional (remote /pnfs reads): pip install fsspec-xrootd xrootd"
