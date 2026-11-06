"""Step 0 candidate (a): pretrained SmolVLA base -> Core ML.

One self-contained Core ML model: SigLIP + 16 SmolVLM2 layers fill the prefix KV cache,
then the action expert runs `num_steps` flow-matching Euler steps (unrolled at trace time).

Fixed shapes for the phone: 1 camera at 512x512, instruction padded to LANG_TOKENS tokens,
state padded to 32, chunk of 50 x 32 (only the first 2 dims would be used for a drive head).

    training/.venv/bin/python training/export/export_smolvla.py [--steps 10] [--lang 16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import convert, parity, record  # noqa: E402

MODEL_ID = "lerobot/smolvla_base"
MODEL_REV = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"  # pinned; recorded in PLAN.md


def patch_static_vision_positions():
    """Unsupported op #1: `bucketize` in SmolVLMVisionEmbeddings (NaViT variable-resolution
    position ids). For a full, fixed-size image every patch is valid and the position ids
    are just arange(num_patches), so replace the dynamic path. Verified equal in main()."""
    from transformers.models.smolvlm import modeling_smolvlm as m

    def forward(self, pixel_values, patch_attention_mask):
        emb = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        assert emb.shape[1] == self.num_patches, "static positions assume a full image_size x image_size input"
        return emb + self.position_embedding.weight[None]

    m.SmolVLMVisionEmbeddings.forward = forward

    # Unsupported op #2: `new_ones` (+ mask building) in SmolVLMVisionTransformer.forward for
    # the all-ones patch mask. A full image needs no vision attention mask at all.
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    def embed_image(self, image):
        vm = self.get_vlm_model().vision_model
        h = vm.embeddings(pixel_values=image.to(vm.dtype), patch_attention_mask=None)
        h = vm.encoder(inputs_embeds=h, attention_mask=None).last_hidden_state
        return self.get_vlm_model().connector(vm.post_layernorm(h))

    SmolVLMWithExpertModel.embed_image = embed_image

    # Unsupported op #4: in-place slice assignment (`res[..., :d] = ...`) in apply_rope.
    import lerobot.policies.smolvla.smolvlm_with_expert as swe

    def apply_rope(x, positions, max_wavelength=10_000):
        d_half = x.shape[-1] // 2
        dtype = x.dtype
        x = x.to(torch.float32)
        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32)
        timescale = max_wavelength**freq_exponents
        radians = (positions[..., None].to(torch.float32) / timescale[None, None, :])[..., None, :]
        sin, cos = torch.sin(radians), torch.cos(radians)
        x1, x2 = x.split(d_half, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)

    swe.apply_rope = apply_rope

    # Not an unsupported op but a silent fp16 failure (Core ML output was all NaN):
    # eager attention masks with torch.finfo(float32).min, which is -inf in fp16, so padded
    # query rows softmax to NaN and the NaN spreads through V. Use an fp16-safe -1e4 and
    # scale q before q.k so scores stay inside fp16 range.
    def eager_attention_forward(self, attention_mask, batch_size, head_dim, q, k, v):
        h, kvh = self.num_attention_heads, self.num_key_value_heads
        g = h // kvh
        L = k.shape[1]
        k = k[:, :, :, None, :].expand(batch_size, L, kvh, g, head_dim).reshape(batch_size, L, h, head_dim)
        v = v[:, :, :, None, :].expand(batch_size, L, kvh, g, head_dim).reshape(batch_size, L, h, head_dim)
        q = q.transpose(1, 2) * (head_dim**-0.5)
        w = torch.matmul(q, k.transpose(1, 2).transpose(2, 3))
        w = torch.where(attention_mask[:, None, :, :], w, torch.tensor(-1e4, dtype=w.dtype))
        p = torch.softmax(w, dim=-1)
        o = torch.matmul(p, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return o.reshape(batch_size, -1, h * head_dim)

    SmolVLMWithExpertModel.eager_attention_forward = eager_attention_forward


class SmolVLAExport(torch.nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.m = flow_model

    def forward(self, image, lang_tokens, lang_mask, state, noise):
        # image: [1,3,512,512] in [0,1] (resize/pad happens in the app); SigLIP wants [-1,1]
        img = image * 2.0 - 1.0
        img_mask = torch.ones(1, dtype=torch.bool)
        return self.m.sample_actions([img], [img_mask], lang_tokens, lang_mask.bool(), state, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10, help="flow-matching Euler steps (base default 10)")
    ap.add_argument("--lang", type=int, default=16, help="padded instruction length in tokens")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(0)  # example image / noise below; recorded in the conversion log

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID, revision=MODEL_REV)
    policy.eval().to("cpu", torch.float32)
    cfg = policy.config
    cfg.num_steps = args.steps
    flow = policy.model
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"SmolVLA params: {n_params / 1e6:.1f}M  chunk={cfg.chunk_size}  steps={cfg.num_steps}")

    tok = flow.vlm_with_expert.processor.tokenizer
    enc = tok("go to the chair\n", padding="max_length", max_length=args.lang, return_tensors="pt")
    ex = (
        torch.rand(1, 3, 512, 512),
        enc["input_ids"].to(torch.int32),
        enc["attention_mask"].to(torch.int32),
        torch.zeros(1, cfg.max_state_dim),
        torch.randn(1, cfg.chunk_size, cfg.max_action_dim),
    )
    wrapper = SmolVLAExport(flow).eval()
    with torch.no_grad():
        orig = wrapper(ex[0], ex[1].long(), ex[2], ex[3], ex[4])
        patch_static_vision_positions()
        ref = wrapper(ex[0], ex[1].long(), ex[2], ex[3], ex[4])
        print(f"static-position patch max diff vs original: {float((orig - ref).abs().max()):.2e}")
        traced = torch.jit.trace(wrapper, (ex[0], ex[1].long(), ex[2], ex[3], ex[4]), check_trace=False)
    print("torch out", tuple(ref.shape))

    name = f"smolvla_base_s{args.steps}_l{args.lang}_{args.precision}"
    inputs = [
        ct.TensorType("image", shape=ex[0].shape, dtype=np.float32),
        ct.TensorType("lang_tokens", shape=ex[1].shape, dtype=np.int32),
        ct.TensorType("lang_mask", shape=ex[2].shape, dtype=np.int32),
        ct.TensorType("robot_state", shape=ex[3].shape, dtype=np.float32),
        ct.TensorType("noise", shape=ex[4].shape, dtype=np.float32),
    ]
    path, log = convert(name, traced, inputs, [ct.TensorType("actions", dtype=np.float32)], precision=args.precision)
    if path is not None:
        feeds = {"image": ex[0].numpy(), "lang_tokens": ex[1].numpy(), "lang_mask": ex[2].numpy(),
                 "robot_state": ex[3].numpy(), "noise": ex[4].numpy()}
        par = parity(path, feeds, ref.numpy(), "actions", "one fixed example (torch.manual_seed(0))")
    record(name, log, model={"id": MODEL_ID, "revision": MODEL_REV}, seeds={"torch.manual_seed": 0},
           settings={"steps": args.steps, "lang_len": args.lang, "image": "1x3x512x512", "chunk": cfg.chunk_size,
                     "max_action_dim": cfg.max_action_dim, "instruction": "go to the chair\\n"},
           params_M=round(n_params / 1e6, 1),
           static_position_patch_max_diff=float((orig - ref).abs().max()),
           **({"parity_vs_torch": par} if path is not None else {}))


if __name__ == "__main__":
    main()
