"""Mac-side PROXY benchmark for Step 0 Core ML models.

This is NOT the phone result. It runs the same .mlpackage on this Mac's Neural Engine /
GPU / CPU to catch broken conversions and get an order-of-magnitude number before the
iPhone run. The real numbers come from the iOS app in app/Bench (see app/README.md).
A Mac M-series chip differs from any iPhone in ANE generation, clocks, memory bandwidth and
thermal envelope, so these numbers must never be quoted as phone numbers.

    training/.venv/bin/python eval/step0/bench_mac.py training/out/<model>.mlpackage --n 50 --trials 5

Every trial is kept: the JSON holds each trial's raw latencies (ms) and stats, plus the median and
spread of the per-trial medians. Each run writes a new timestamped file (nothing is overwritten), with
the date, git commit, command line, tool versions, input/output shapes and the input seed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np

REPO = Path(__file__).resolve().parents[2]
UNITS = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
}
INPUT_SEED = 0
WARMUP = 3


def fixed_inputs(spec) -> dict:
    """Float inputs uniform [0,1) from numpy default_rng(INPUT_SEED); int inputs all ones."""
    rng = np.random.default_rng(INPUT_SEED)
    feeds = {}
    for inp in spec.description.input:
        shape = tuple(inp.type.multiArrayType.shape)
        dt = inp.type.multiArrayType.dataType
        if dt == ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32:
            feeds[inp.name] = np.ones(shape, dtype=np.int32)
        else:
            feeds[inp.name] = rng.random(shape, dtype=np.float32)
    return feeds


def io_shapes(spec) -> dict:
    def shp(f):
        return list(f.type.multiArrayType.shape)
    return {"inputs": {i.name: shp(i) for i in spec.description.input},
            "outputs": {o.name: shp(o) for o in spec.description.output}}


def stats(a: np.ndarray) -> dict:
    return {"n": int(len(a)), "median_ms": round(float(np.median(a)), 2),
            "p5_ms": round(float(np.percentile(a, 5)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
            "min_ms": round(float(a.min()), 2), "max_ms": round(float(a.max()), 2),
            "mean_ms": round(float(a.mean()), 2), "std_ms": round(float(a.std()), 2)}


def provenance() -> dict:
    def run(*cmd):
        r = subprocess.run(list(cmd), capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": run("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
        "host": platform.platform(),
        "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
        "versions": {"coremltools": ct.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "power_source": run("pmset", "-g", "batt"),
    }


def bench_unit(path: Path, uname: str, n: int, seconds: float) -> dict:
    t0 = time.time()
    m = ct.models.MLModel(str(path), compute_units=UNITS[uname])
    load_s = time.time() - t0
    feeds = fixed_inputs(m.get_spec())
    first_t = time.time()
    m.predict(feeds)
    first_s = time.time() - first_t
    for _ in range(WARMUP):
        m.predict(feeds)
    lat = []
    end = time.time() + seconds
    while (len(lat) < n) if seconds <= 0 else (time.time() < end):
        t = time.perf_counter()
        m.predict(feeds)
        lat.append((time.perf_counter() - t) * 1000)
    a = np.array(lat)
    r = {"load_s": round(load_s, 2), "first_predict_s": round(first_s, 2), **stats(a),
         "latencies_ms": [round(x, 3) for x in lat]}
    del m
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=50, help="timed predictions per trial")
    ap.add_argument("--trials", type=int, default=5, help="independent trials (fresh model load each)")
    ap.add_argument("--units", default="CPU_AND_NE,ALL,CPU_AND_GPU")
    ap.add_argument("--seconds", type=float, default=0, help="sustained run length per trial instead of --n")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "step0" / "mac_proxy")  # gitignored
    args = ap.parse_args()

    path = Path(args.model)
    spec = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY).get_spec()
    results = {"model": path.name, "note": "Mac proxy only; NOT phone numbers. Fixed synthetic inputs.",
               "provenance": provenance(), "io": io_shapes(spec),
               "settings": {"n": args.n, "trials": args.trials, "seconds": args.seconds, "warmup": WARMUP,
                            "input_seed": INPUT_SEED, "inputs": "float: numpy default_rng(seed).random; int32: ones",
                            "unit_order": "units interleaved within each trial"},
               "runs": {u: {"trials": []} for u in args.units.split(",")}}
    del spec
    # Interleave units inside each trial so slow drift (thermal, background load) hits every unit alike.
    for trial in range(args.trials):
        for uname in results["runs"]:
            r = bench_unit(path, uname, args.n, args.seconds)
            r["trial"] = trial
            results["runs"][uname]["trials"].append(r)
            print(uname, trial, {k: v for k, v in r.items() if k != "latencies_ms"}, flush=True)
    for uname, u in results["runs"].items():
        meds = np.array([t["median_ms"] for t in u["trials"]])
        u["across_trials"] = {"median_of_medians_ms": round(float(np.median(meds)), 2),
                              "min_median_ms": round(float(meds.min()), 2), "max_median_ms": round(float(meds.max()), 2),
                              "pooled": stats(np.concatenate([t["latencies_ms"] for t in u["trials"]]))}
    # Whole-process peak over all units and trials (not per unit; the ANE weight copy is not charged here).
    results["peak_rss_mb_process_all_units"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0)
    stamp = results["provenance"]["date_utc"].replace(":", "-")
    out = args.out_dir / f"{path.stem}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
