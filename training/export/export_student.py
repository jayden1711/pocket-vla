"""Step 0 candidate (b): ~76M student VLA (random weights) -> Core ML.

Writes three packages to training/out/ (git-ignored) and a conversion log per package to
results/step0/conversion/ (git-ignored):

    student_full_<prec>     image + lang tokens + state -> actions [1,16,2]   (single model, worst case)
    student_text_<prec>     lang tokens -> text features [1,16,512]          (run once per instruction)
    student_policy_<prec>   image + text features + state -> actions          (run per frame)

Random weights: this is a latency/conversion check only. The weights are seeded so reruns are identical.

    training/.venv/bin/python training/export/export_student.py [--precision fp16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from common import convert, parity, record  # noqa: E402
from student import StudentConfig, StudentVLA, param_counts  # noqa: E402


class TextOnly(torch.nn.Module):
    def __init__(self, m: StudentVLA):
        super().__init__()
        self.text = m.text

    def forward(self, lang_tokens, lang_mask):
        return self.text(lang_tokens, lang_mask)


class PolicyOnly(torch.nn.Module):
    def __init__(self, m: StudentVLA):
        super().__init__()
        self.policy = m.policy

    def forward(self, image, text_feats, lang_mask, robot_state):
        return self.policy.forward_with_text(image, text_feats, lang_mask, robot_state)


def check(path: Path, feeds: dict, ref: torch.Tensor, out_name: str) -> dict:
    return parity(path, {k: v.numpy() for k, v in feeds.items()}, ref.numpy(), out_name,
                  "one fixed example (torch.Generator().manual_seed(1))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()

    torch.manual_seed(0)
    c = StudentConfig()
    model = StudentVLA(c).eval()
    print("student params:", param_counts(model))

    g = torch.Generator().manual_seed(1)
    image = torch.rand(1, 3, c.image_size, c.image_size, generator=g)
    tokens = torch.randint(0, c.vocab_size, (1, c.lang_len), generator=g, dtype=torch.int32)
    mask = torch.zeros(1, c.lang_len, dtype=torch.int32)
    mask[:, :6] = 1  # "go to the red chair" ~ 6 tokens, rest padding
    state = torch.zeros(1, c.state_dim)

    with torch.no_grad():
        ref_full = model(image, tokens.long(), mask, state)
        text_feats = model.text(tokens.long(), mask)
        ref_policy = model.policy.forward_with_text(image, text_feats, mask, state)
        assert torch.allclose(ref_full, ref_policy), "split model must equal the full model"
        # Core ML gather wants int32 indices; trace with int32 so the converter keeps them.
        full_tr = torch.jit.trace(model, (image, tokens, mask, state), check_trace=False)
        text_tr = torch.jit.trace(TextOnly(model).eval(), (tokens, mask), check_trace=False)
        pol_tr = torch.jit.trace(PolicyOnly(model).eval(), (image, text_feats, mask, state), check_trace=False)

    T = ct.TensorType
    img_in = T("image", shape=image.shape, dtype=np.float32)
    tok_in = T("lang_tokens", shape=tokens.shape, dtype=np.int32)
    mask_in = T("lang_mask", shape=mask.shape, dtype=np.int32)
    st_in = T("robot_state", shape=state.shape, dtype=np.float32)  # "state" is reserved by Core ML
    tf_in = T("text_feats", shape=text_feats.shape, dtype=np.float32)
    actions = [T("actions", dtype=np.float32)]
    p = args.precision

    jobs = [
        (f"student_full_{p}", full_tr, [img_in, tok_in, mask_in, st_in], actions,
         {"image": image, "lang_tokens": tokens, "lang_mask": mask, "robot_state": state}, ref_full, "actions"),
        (f"student_text_{p}", text_tr, [tok_in, mask_in], [T("text_feats", dtype=np.float32)],
         {"lang_tokens": tokens, "lang_mask": mask}, text_feats, "text_feats"),
        (f"student_policy_{p}", pol_tr, [img_in, tf_in, mask_in, st_in], actions,
         {"image": image, "text_feats": text_feats, "lang_mask": mask, "robot_state": state}, ref_policy, "actions"),
    ]
    for name, traced, inputs, outputs, feeds, ref, out_name in jobs:
        path, log = convert(name, traced, inputs, outputs, precision=p)
        extra = {"model": "student (random weights, untrained; training/models/student.py)",
                 "seeds": {"weights": "torch.manual_seed(0)", "example_inputs": "torch.Generator().manual_seed(1)"},
                 "config": vars(c), "params": param_counts(model)}
        if path is not None:
            extra["parity_vs_torch"] = check(path, feeds, ref, out_name)
        record(name, log, **extra)


if __name__ == "__main__":
    main()
