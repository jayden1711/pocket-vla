"""Step 0 candidate (c): OmniVLA-edge (MIT, real weights) -> Core ML, language-goal mode only.

STATUS: written 2026-11-06 overnight, NOT RUN YET (executing the downloaded upstream model file and
loading its checkpoint was held for the owner's approval). Expect small fixes on the first run.

Writes to training/out/ (git-ignored), logs to results/step0/conversion/ (git-ignored):

    omnivla_edge_lang_<prec>   6 past frames 96x96 + current frame 224x224 + CLIP text feature [1,512]
                               -> actions [1,8,4] (cumulative x, y waypoints, cos, sin of heading), dist [1,1]
    clip_b32_text_<prec>       CLIP ViT-B/32 token ids [1,77] -> text feature [1,512] (once per instruction)

OmniVLA's own code is not vendored. Point --omnivla-src at a checkout of NHirose/OmniVLA at the pinned
commit (THIRD_PARTY.md); only inference/model_omnivla_edge.py is imported (needs efficientnet_pytorch 0.7.1).

    curl -sSL https://codeload.github.com/NHirose/OmniVLA/tar.gz/5182600cb4a9 | tar -xz -C /tmp
    training/.venv/bin/python training/export/export_omnivla_edge.py --omnivla-src /tmp/OmniVLA-5182600cb4a9

Language-goal mode is modality 7: goal pose, satellite map and goal image tokens are masked out of the
decoder's attention and its average pool, so their encoders cannot change the output. The export drops
those three tokens (and their two EfficientNets) and keeps the remaining tokens at their original
positions. The script checks this pruned forward against the original forward before converting.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import convert  # noqa: E402

EDGE_REPO, EDGE_REV = "NHirose/omnivla-edge", "b1361b7e24f101edea795a98b00a826b61a97394"
CLIP_REPO, CLIP_REV = "openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
LANG_MODALITY = 7
# run_omnivla_edge.py model_params for the released checkpoint
EDGE_ARGS = dict(context_size=5, len_traj_pred=8, learn_angle=True, obs_encoder="efficientnet-b0",
                 obs_encoding_size=1024, late_fusion=False, mha_num_attention_heads=4,
                 mha_num_attention_layers=4, mha_ff_dim_factor=4)


def load_edge(src: Path):
    spec = importlib.util.spec_from_file_location("model_omnivla_edge", src / "inference" / "model_omnivla_edge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(EDGE_REPO, "omnivla-edge.pth", revision=EDGE_REV)
    model = mod.OmniVLA_edge(**EDGE_ARGS)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for enc in (model.obs_encoder, model.goal_encoder, model.goal_encoder_img):
        enc.set_swish(memory_efficient=False)  # custom autograd swish is not traceable
    return model


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_frames(x: torch.Tensor) -> torch.Tensor:
    """[b, 3k, H, W] in [0,1] -> ImageNet-normalised, per 3-channel frame."""
    b, c, h, w = x.shape
    return ((x.view(b, c // 3, 3, h, w) - MEAN[:, None]) / STD[:, None]).view(b, c, h, w)


class EdgeLanguagePolicy(torch.nn.Module):
    """OmniVLA-edge forward for modality 7 with masked branches removed. Inputs in [0,1]."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        n = m.context_size + 5
        mask = m.all_masks[LANG_MODALITY]  # True = token masked out
        keep = [i for i in range(n) if not bool(mask[i])]
        assert keep == list(range(m.context_size + 1)) + [n - 1], keep  # 6 obs tokens + language token
        self.register_buffer("pos", m.decoder.positional_encoding.pos_enc[:, keep, :].clone(), persistent=False)
        self.pool_scale = float(m.avg_pool_mask[LANG_MODALITY][keep[0]]) * len(keep) / n  # mean over n positions
        self.modality = float(LANG_MODALITY)

    def forward(self, obs_frames, current_img, text_feat):
        m = self.m
        obs = normalize_frames(obs_frames)
        cur = normalize_frames(current_img)
        lan = m.compress_goal_enc_lan(m.film_model(cur, text_feat).flatten(start_dim=1)).unsqueeze(1)
        frames = torch.cat(torch.split(obs, 3, dim=1), dim=0)  # [6, 3, 96, 96]
        z = m.obs_encoder._avg_pooling(m.obs_encoder.extract_features(frames)).flatten(start_dim=1)
        z = m.compress_obs_enc(z).reshape(m.context_size + 1, -1, m.obs_encoding_size).transpose(0, 1)
        x = torch.cat([z, lan], dim=1) + self.pos
        for layer in m.decoder.sa_decoder.layers:
            x = layer(x)
        if m.decoder.sa_decoder.norm is not None:
            x = m.decoder.sa_decoder.norm(x)
        # upstream: mean over all n positions of x * avg_pool_mask (n/len(keep) on kept tokens, 0 on masked)
        x = x.mean(dim=1) * self.pool_scale
        x = torch.cat([x, torch.full_like(x[:, :1], self.modality)], dim=1)
        for lin in m.decoder.output_layers:
            x = F.relu(lin(x))
        a = m.action_predictor(x).reshape(x.shape[0], m.len_trajectory_pred, m.num_action_params)
        actions = torch.cat([torch.cumsum(a[:, :, :2], dim=1), F.normalize(a[:, :, 2:], dim=-1)], dim=-1)
        return actions, m.dist_predictor(x)


def original_forward(m, obs_frames, current_img, text_feat):
    """The unmodified OmniVLA_edge.forward with dummy pose/map/goal inputs (as run_omnivla_edge.py does)."""
    obs = normalize_frames(obs_frames)
    cur = normalize_frames(current_img)
    zeros96 = normalize_frames(torch.zeros(1, 6, 96, 96))
    map_images = torch.cat([zeros96, obs[:, -3:]], dim=1)
    goal_img = normalize_frames(torch.rand(1, 3, 96, 96))
    goal_pose = torch.randn(1, 4)
    get_device = torch.Tensor.get_device
    torch.Tensor.get_device = lambda self: "cpu"  # upstream assumes CUDA (get_device() is -1 on CPU)
    try:
        a, d, _ = m(obs, goal_pose, map_images, goal_img, torch.tensor([LANG_MODALITY]), text_feat, cur)
    finally:
        torch.Tensor.get_device = get_device
    return a, d


class ClipText(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, input_ids):
        return self.clip(input_ids=input_ids).text_embeds  # = openai clip encode_text (unnormalised)


def report(path, feeds, refs, names):
    ml = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({k: v.numpy() for k, v in feeds.items()})
    for n, r in zip(names, refs):
        err = float(np.abs(out[n] - r.numpy()).max())
        print(f"  {path.name}:{n} max |coreml - torch| = {err:.4f} (scale {float(r.abs().max()):.3f}), "
              f"finite={bool(np.isfinite(out[n]).all())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omnivla-src", type=Path, required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)  # traceable attention path

    m = load_edge(args.omnivla_src)
    total = sum(p.numel() for p in m.parameters()) / 1e6
    pol = EdgeLanguagePolicy(m).eval()
    used = sum(p.numel() for n, p in m.named_parameters()
               if not n.startswith(("goal_encoder.", "goal_encoder_img.", "local_goal.", "compress_obs_enc_map",
                                    "compress_goal_enc_img"))) / 1e6
    print(f"OmniVLA-edge params: {total:.1f}M total, {used:.1f}M used in language mode")

    g = torch.Generator().manual_seed(0)
    obs = torch.rand(1, 18, 96, 96, generator=g)
    cur = torch.rand(1, 3, 224, 224, generator=g)
    txt = torch.randn(1, 512, generator=g) * 0.3
    with torch.no_grad():
        a0, d0 = original_forward(m, obs, cur, txt)
        a1, d1 = pol(obs, cur, txt)
        print(f"pruned vs original: actions {float((a0 - a1).abs().max()):.2e}, dist {float((d0 - d1).abs().max()):.2e}")
        assert torch.allclose(a0, a1, atol=1e-4) and torch.allclose(d0, d1, atol=1e-3)
        traced = torch.jit.trace(pol, (obs, cur, txt), check_trace=False)

    T = ct.TensorType
    p = args.precision
    path, _ = convert(f"omnivla_edge_lang_{p}", traced,
                      [T("obs_frames", shape=obs.shape, dtype=np.float32),
                       T("current_img", shape=cur.shape, dtype=np.float32),
                       T("text_feat", shape=txt.shape, dtype=np.float32)],
                      [T("actions", dtype=np.float32), T("dist", dtype=np.float32)], precision=p)
    if path:
        report(path, {"obs_frames": obs, "current_img": cur, "text_feat": txt}, [a1, d1], ["actions", "dist"])

    if args.skip_text:
        return
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP_REPO, revision=CLIP_REV)
    clip = CLIPTextModelWithProjection.from_pretrained(CLIP_REPO, revision=CLIP_REV, attn_implementation="eager")
    clip.eval()
    ids = tok(["blue trash bin"], padding="max_length", max_length=77, return_tensors="pt")["input_ids"].to(torch.int32)
    ct_mod = ClipText(clip).eval()
    with torch.no_grad():
        ref = ct_mod(ids)
        traced_t = torch.jit.trace(ct_mod, (ids,), check_trace=False)
    path, _ = convert(f"clip_b32_text_{p}", traced_t, [T("input_ids", shape=ids.shape, dtype=np.int32)],
                      [T("text_feat", dtype=np.float32)], precision=p)
    if path:
        report(path, {"input_ids": ids}, [ref], ["text_feat"])


if __name__ == "__main__":
    main()
