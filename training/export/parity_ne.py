"""Core ML prediction in a fresh process (no torch loaded), for the parity check on a given compute unit.

Loading an .mlpackage with CPU_AND_NE inside the export process (torch 2.15 + coremltools 9.0 loaded)
segfaulted on this Mac on 2026-11-06; in a process without torch it works. So common.parity() calls this.

    python parity_ne.py <model.mlpackage> <feeds.npz> <ref.npy> <output_name> <COMPUTE_UNIT>   -> JSON on stdout
"""

import json
import sys

import coremltools as ct
import numpy as np

path, feeds_npz, ref_npy, out_name, unit = sys.argv[1:6]
feeds = dict(np.load(feeds_npz))
ref = np.load(ref_npy)
out = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, unit)).predict(feeds)[out_name]
print(json.dumps({"max_abs_err": float(np.abs(out - ref).max()), "mean_abs_err": float(np.abs(out - ref).mean()),
                  "finite": bool(np.isfinite(out).all())}))
