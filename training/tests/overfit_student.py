"""Trainability sanity check for the student (SYNTHETIC data, not robot data, not a task result).

Question it answers: can the student architecture + loss fit a tiny synthetic dataset at all, and does the
language path carry information? It does NOT say anything about driving, generalisation or real accuracy.

Dataset: K random images; for each image, two different random instructions with two different random target
chunks in [-0.9, 0.9]. A model that ignores the instruction cannot get below the "image-only" floor (it must
predict the mean of the two targets per image), so the floor is reported next to the final loss.

Each seed is a full independent run; every seed's loss curve is kept, nothing is dropped.

    training/.venv/bin/python training/tests/overfit_student.py --seeds 0,1,2 --steps 300
    -> results/student/overfit_<date>.json (git-ignored)
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "models"))
from student import StudentConfig, StudentVLA, param_counts  # noqa: E402


def make_data(c: StudentConfig, k: int, seed: int):
    g = torch.Generator().manual_seed(1000 + seed)
    img = torch.rand(k, 3, c.image_size, c.image_size, generator=g).repeat_interleave(2, 0)
    tok = torch.randint(1, c.vocab_size, (2 * k, c.lang_len), generator=g)
    mask = torch.zeros(2 * k, c.lang_len, dtype=torch.int32)
    mask[:, :6] = 1
    state = torch.zeros(2 * k, c.state_dim)
    y = (torch.rand(2 * k, c.chunk, c.action_dim, generator=g) * 1.8 - 0.9)
    pair_mean = y.view(k, 2, c.chunk, c.action_dim).mean(1, keepdim=True).expand(-1, 2, -1, -1).reshape_as(y)
    return (img, tok, mask, state), y, float(F.mse_loss(pair_mean, y))


def run(seed: int, steps: int, k: int, lr: float, device: str) -> dict:
    torch.manual_seed(seed)
    c = StudentConfig()
    m = StudentVLA(c).to(device).train()
    x, y, image_only_floor = make_data(c, k, seed)
    x = tuple(t.to(device) for t in x)
    y = y.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    mean_floor = float(F.mse_loss(y.mean(0, keepdim=True).expand_as(y), y))
    losses = []
    t0 = time.time()
    for _ in range(steps):
        loss = F.mse_loss(m(*x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(float(loss), 6))
    m.eval()
    with torch.no_grad():
        final = float(F.mse_loss(m(*x), y))
    return {"seed": seed, "losses": losses, "final_eval_mse": final, "constant_mean_floor_mse": mean_floor,
            "image_only_floor_mse": image_only_floor, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--k", type=int, default=4, help="distinct images (each with 2 instructions)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "student")
    args = ap.parse_args()

    def git(*a):
        r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    out = {"what": "SYNTHETIC overfit sanity check; not robot data, not a task metric",
           "provenance": {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                          "git_commit": git("rev-parse", "HEAD"),
                          "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                          "argv": sys.argv, "host": platform.platform(),
                          "versions": {"torch": torch.__version__, "python": platform.python_version()}},
           "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "student_config": vars(StudentConfig()), "params": param_counts(StudentVLA(StudentConfig())),
           "runs": []}
    for s in [int(v) for v in args.seeds.split(",")]:
        r = run(s, args.steps, args.k, args.lr, args.device)
        print(f"seed {s}: first {r['losses'][0]:.4f} final {r['final_eval_mse']:.4f} "
              f"image-only floor {r['image_only_floor_mse']:.4f} ({r['seconds']} s)", flush=True)
        out["runs"].append(r)
    stamp = out["provenance"]["date_utc"].replace(":", "-")
    path = args.out_dir / f"overfit_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("->", path)


if __name__ == "__main__":
    main()
