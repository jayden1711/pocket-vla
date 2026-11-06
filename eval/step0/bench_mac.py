"""Mac-side PROXY benchmark for Step 0 Core ML models.

This is NOT the phone result. It runs the same .mlpackage on this Mac's Neural Engine /
GPU / CPU to catch broken conversions and get an order-of-magnitude number before the
iPhone run. The real numbers come from the iOS app in app/Bench (see app/README.md).

    training/.venv/bin/python eval/step0/bench_mac.py training/out/<model>.mlpackage --n 50
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np

UNITS = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
}


def fixed_inputs(spec) -> dict:
    rng = np.random.default_rng(0)
    feeds = {}
    for inp in spec.description.input:
        shape = tuple(inp.type.multiArrayType.shape)
        dt = inp.type.multiArrayType.dataType
        if dt == ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32:
            feeds[inp.name] = np.ones(shape, dtype=np.int32)
        else:
            feeds[inp.name] = rng.random(shape, dtype=np.float32)
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--units", default="CPU_AND_NE,ALL,CPU_AND_GPU")
    ap.add_argument("--seconds", type=float, default=0, help="sustained run length instead of --n")
    args = ap.parse_args()

    path = Path(args.model)
    results = {"model": path.name, "host": platform.platform(),
               "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
               "note": "Mac proxy only; not phone numbers", "runs": {}}
    for uname in args.units.split(","):
        t0 = time.time()
        m = ct.models.MLModel(str(path), compute_units=UNITS[uname])
        load_s = time.time() - t0
        feeds = fixed_inputs(m.get_spec())
        first_t = time.time()
        m.predict(feeds)
        first_s = time.time() - first_t
        for _ in range(3):
            m.predict(feeds)
        lat = []
        end = time.time() + args.seconds
        while (len(lat) < args.n) if args.seconds <= 0 else (time.time() < end):
            t = time.perf_counter()
            m.predict(feeds)
            lat.append((time.perf_counter() - t) * 1000)
        a = np.array(lat)
        r = {"n": len(a), "load_s": round(load_s, 2), "first_predict_s": round(first_s, 2),
             "median_ms": round(float(np.median(a)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
             "hz": round(1000 / float(np.median(a)), 2),
             "peak_rss_mb_process": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0)}
        results["runs"][uname] = r
        print(uname, r, flush=True)
        del m
    out = Path(__file__).resolve().parents[2] / "results" / "step0" / "mac_proxy" / f"{path.stem}.json"  # gitignored
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
