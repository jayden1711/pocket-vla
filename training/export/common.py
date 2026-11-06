"""Shared Core ML conversion + logging helpers for Step 0 exports.

Every conversion writes a JSON log under results/step0/conversion/ (git-ignored) with: tool versions, the
ops in the converted program, ops the converter could not handle (if conversion failed),
and the per-op compute-device plan (ANE / GPU / CPU) as reported by Core ML on this Mac.
Ops that fall off the Neural Engine are the ones to watch on the phone.
"""

from __future__ import annotations

import collections
import io
import json
import logging
import datetime
import platform
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import coremltools as ct
import torch

# --- converter patches (each one is an "unsupported op" entry in PLAN.md) -------------
from coremltools.converters.mil import Builder as mb  # noqa: E402
from coremltools.converters.mil.frontend.torch.ops import _get_inputs  # noqa: E402
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op  # noqa: E402
from coremltools.converters.mil.mil import types  # noqa: E402


@register_torch_op(override=True)
def reciprocal(context, node):
    """torch.reciprocal on an integer tensor -> Core ML `inverse` only accepts floats."""
    x = _get_inputs(context, node, expected=1)[0]
    if not types.is_float(x.dtype):
        x = mb.cast(x=x, dtype="fp32")
    context.add(mb.inverse(x=x, name=node.name))


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "training" / "out"  # gitignored: mlpackages, checkpoints
LOG_DIR = REPO / "results" / "step0" / "conversion"  # gitignored until approved


def provenance() -> dict:
    """Who/what/when for a result file: UTC date, git commit (+ dirty flag), exact command line."""
    def git(*a):
        r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
    }


def versions() -> dict:
    import transformers

    return {
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "host": platform.platform(),
    }


def convert(name: str, traced, inputs, outputs, *, precision="fp16", target=ct.target.iOS18) -> tuple[Path | None, dict]:
    """Convert a traced module to an .mlpackage. Never raises; failures are logged."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {"name": name, "provenance": provenance(), "versions": versions(), "precision": precision, "target": str(target)}
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logging.getLogger("coremltools").addHandler(handler)
    t0 = time.time()
    path = OUT_DIR / f"{name}.mlpackage"
    try:
        with redirect_stderr(buf), redirect_stdout(buf):
            mlmodel = ct.convert(
                traced,
                inputs=inputs,
                outputs=outputs,
                convert_to="mlprogram",
                minimum_deployment_target=target,
                compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
            )
        mlmodel.save(str(path))
        log["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - we want every failure in the log
        log["status"] = "failed"
        log["error"] = f"{type(e).__name__}: {e}"
        log["traceback"] = traceback.format_exc()[-4000:]
        path = None
    finally:
        logging.getLogger("coremltools").removeHandler(handler)
    log["convert_seconds"] = round(time.time() - t0, 1)
    warn_lines = [ln for ln in buf.getvalue().splitlines() if any(k in ln.lower() for k in ("warn", "not supported", "unsupported", "fallback", "error"))]
    log["converter_warnings"] = sorted(set(warn_lines))[:200]
    if path is not None:
        log.update(inspect_package(path))
    write_log(name, log)
    return path, log


def inspect_package(path: Path) -> dict:
    """Op histogram + per-op compute device plan from Core ML (macOS only)."""
    info: dict = {"mlpackage": str(path.relative_to(REPO)), "size_mb": round(dir_size(path) / 1e6, 1)}
    try:
        mlm = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)  # keep alive: owns the .mlmodelc
        compiled = mlm.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.ALL)
        program = plan.model_structure.program
        ops = collections.Counter()
        devices = collections.Counter()
        off_ane = collections.Counter()
        cost_by_dev = collections.Counter()
        for fn in program.functions.values():
            for op in fn.block.operations:
                ops[op.operator_name] += 1
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                dev = type(usage.preferred_compute_device).__name__ if usage else "none"
                devices[dev] += 1
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                w = cost.weight if cost else 0.0
                cost_by_dev[dev] += w
                if usage and "NeuralEngine" not in dev and op.operator_name not in ("const",):
                    off_ane[f"{op.operator_name} -> {dev}"] += 1
        info["op_histogram"] = dict(ops.most_common())
        info["ops_by_preferred_device"] = dict(devices)
        info["estimated_cost_share_by_device"] = {k: round(v, 3) for k, v in cost_by_dev.items()}
        info["ops_off_neural_engine"] = dict(off_ane.most_common())
    except Exception as e:  # noqa: BLE001
        info["compute_plan_error"] = f"{type(e).__name__}: {e}"
    return info


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def parity(path: Path, feeds: dict, ref, out_name: str, inputs_note: str) -> dict:
    """Core ML vs torch (fp32 CPU reference) on one fixed input, on the CPU path and the Neural Engine path.
    Each Core ML run happens in a fresh process without torch (see parity_ne.py for why)."""
    import tempfile

    import numpy as np

    ref = np.asarray(ref)
    res = {"ref_scale": float(np.abs(ref).max()), "inputs": inputs_note, "reference": "torch fp32 CPU"}
    with tempfile.TemporaryDirectory() as td:
        np.savez(f"{td}/feeds.npz", **{k: np.asarray(v) for k, v in feeds.items()})
        np.save(f"{td}/ref.npy", ref)
        for label, unit in (("cpu_only", "CPU_ONLY"), ("cpu_and_ne", "CPU_AND_NE")):
            r = subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "parity_ne.py"), str(path),
                                f"{td}/feeds.npz", f"{td}/ref.npy", out_name, unit], capture_output=True, text=True)
            try:
                res[label] = json.loads(r.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                res[label] = {"error": f"exit {r.returncode}", "stderr_tail": r.stderr[-1500:]}
            print(f"  {path.name} [{label}]: {res[label]}")
    return res


def record(name: str, log: dict, **fields) -> None:
    """Add fields (model ID/revision, seeds, export settings, Core ML vs torch check) to the conversion log
    and rewrite it, so every claim about a conversion lives in a results file."""
    log.update(fields)
    write_log(name, log)


def write_log(name: str, log: dict) -> None:
    """Latest log at <name>.json; every attempt (including failed ones) also kept under history/."""
    text = json.dumps(log, indent=2, default=str)
    (LOG_DIR / f"{name}.json").write_text(text)
    stamp = log.get("provenance", {}).get("date_utc", "undated").replace(":", "-")
    (LOG_DIR / "history").mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "history" / f"{name}_{stamp}.json").write_text(text)
    print(f"[{name}] {log['status']}  ->  {LOG_DIR / (name + '.json')}")
