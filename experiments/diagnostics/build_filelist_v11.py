#!/usr/bin/env python3
"""Build the v11 filelist from the SBND SAM definition and dedup against the
existing full filelist.

sam_web_client on cvmfs is Python2 (fails on this host), so we hit the SBND SAM
HTTP API directly (samsbnd.fnal.gov:8483, unauthenticated GET):
  1. /definitions/name/<def>/files/list?format=plain   -> file names
  2. /files/locations?file_name=...(bulk)               -> dCache dirs
Full /pnfs path = <location dir minus 'dcache:'>/<filename>.

Outputs (study dir):
  filelist_v11_defn.txt   all files in the definition (/pnfs paths)
  filelist_v11_new.txt    genuinely-new files (basename not in filelist_drift_full.txt)
Requires network -> run with the Bash sandbox disabled.
"""
import os
import sys
import time
import json
import ssl
import urllib.request
import urllib.parse

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

DEF = ("data_SBND2026A_Gen2_Run2_crossingmu_2026Summer_DENT_Study"
       "_v10_14_02_04_histreco2_sbnd")
BASE = "https://samsbnd.fnal.gov:8483/sam/sbnd/api"
HERE = "/exp/sbnd/app/users/gputnam/cathode-study"
EXISTING = os.path.join(HERE, "filelist_drift_full.txt")
OUT_DEFN = os.path.join(HERE, "filelist_v11_defn.txt")
OUT_NEW = os.path.join(HERE, "filelist_v11_new.txt")
BATCH = 50


def get_text(url, **params):
    if params:
        url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, context=_SSL, timeout=120) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as e:  # noqa
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
            sys.stderr.write(f"  retry {attempt+1} on {url[:80]}: {e}\n")


def main():
    print(f"definition: {DEF}")
    txt = get_text(f"{BASE}/definitions/name/{DEF}/files/list", format="plain")
    names = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    print(f"files in definition: {len(names):,}")

    # bulk-resolve locations
    paths = []
    missing = []
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        d = json.loads(get_text(f"{BASE}/files/locations", file_name=chunk))
        for n in chunk:
            locs = d.get(n) or []
            # prefer a disk/dcache location
            loc = None
            for L in locs:
                fp = L.get("full_path") or L.get("location") or ""
                if fp.startswith("dcache:") or fp.startswith("/pnfs"):
                    loc = fp
                    break
            if loc is None and locs:
                loc = locs[0].get("full_path") or locs[0].get("location")
            if not loc:
                missing.append(n)
                continue
            d_dir = loc.split(":", 1)[1] if ":" in loc else loc
            paths.append(os.path.join(d_dir, n))
        if (i // BATCH) % 20 == 0:
            print(f"  resolved {min(i+BATCH, len(names)):,}/{len(names):,}",
                  flush=True)

    paths.sort()
    with open(OUT_DEFN, "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"wrote {OUT_DEFN}: {len(paths):,} paths ({len(missing)} unlocated)")

    # dedup by basename vs the existing full filelist
    existing = set()
    with open(EXISTING) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                existing.add(os.path.basename(ln))
    new = [p for p in paths if os.path.basename(p) not in existing]
    with open(OUT_NEW, "w") as f:
        f.write("\n".join(new) + "\n")
    print(f"existing full filelist basenames: {len(existing):,}")
    print(f"wrote {OUT_NEW}: {len(new):,} genuinely-new files "
          f"({len(paths)-len(new):,} overlapped)")


if __name__ == "__main__":
    main()
