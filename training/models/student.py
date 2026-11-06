"""Step 0 candidate (b): a ~76M-parameter student VLA for the phone.

Layout (all plain PyTorch, no third-party model code, so nothing to pin):

    image [1,3,224,224] in [0,1] --ViT-S/16 (12 x 384, 196 patch tokens)--> proj 384->512 --+
    lang tokens [1,L] -------------text encoder (L layers x 384)-----------> proj 384->512 --+--> fusion
    robot state [1,S] ----------------------------------------linear S->512 (1 token) -------+    (10 x 512)
    16 learned action queries ---------------------------------------------------------------+
                                                                                   action queries --> linear 512->2
                                                                                   -> actions [1,16,2]

The action head is a direct regression of a 16-step chunk of 2 wheel commands (ACT-style, one forward
pass), not flow matching (the flow-step sweep in CRITIQUES C1 would measure the cost difference). Weights are random
for Step 0 (latency only); training/distillation is Step 4.

The text encoder can be exported on its own (`TextEncoder`) and run once per instruction, so the per-frame
policy (`StudentPolicy.forward_with_text`) only sees cached text features. `StudentVLA.forward` runs both.

Phone-friendly choices: fixed shapes; no in-place ops; attention masks use an fp16-safe -1e4 (finfo.min
becomes -inf in fp16 and turns padded rows into NaN, see PLAN.md); GELU with tanh approximation;
LayerNorm only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

MASK_NEG = -1e4  # fp16-safe "minus infinity" for attention masks


@dataclass
class StudentConfig:
    image_size: int = 224
    patch: int = 16
    vit_dim: int = 384  # ViT-S
    vit_depth: int = 12
    vit_heads: int = 6
    vocab_size: int = 49280  # SmolVLM2 tokenizer (Apache-2.0, already pinned in THIRD_PARTY.md)
    lang_len: int = 16
    text_dim: int = 384
    text_depth: int = 2
    text_heads: int = 6
    state_dim: int = 8  # e.g. wheel speeds from encoders, last command; zero-padded
    fusion_dim: int = 512
    fusion_depth: int = 10
    fusion_heads: int = 8
    chunk: int = 16
    action_dim: int = 2  # left, right wheel command in [-1, 1]
    mlp_ratio: float = 4.0


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # scale q before q.k to keep scores in fp16 range
        w = torch.matmul(q * (self.head_dim**-0.5), k.transpose(-2, -1))
        if bias is not None:
            w = w + bias  # [b,1,1,n] additive key mask
        p = torch.softmax(w, dim=-1)
        o = torch.matmul(p, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), bias)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


def key_bias(mask: torch.Tensor) -> torch.Tensor:
    """[b,n] 0/1 mask -> [b,1,1,n] additive bias (0 keep, -1e4 drop)."""
    return ((1.0 - mask.to(torch.float32)) * MASK_NEG)[:, None, None, :]


class ImageEncoder(nn.Module):
    """ViT-S/16 with learned positions; returns all patch tokens (no CLS)."""

    def __init__(self, c: StudentConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, c.vit_dim, kernel_size=c.patch, stride=c.patch)
        n = (c.image_size // c.patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, c.vit_dim) * 0.02)
        self.blocks = nn.ModuleList(Block(c.vit_dim, c.vit_heads, c.mlp_ratio) for _ in range(c.vit_depth))
        self.norm = nn.LayerNorm(c.vit_dim, eps=1e-6)
        # ImageNet normalisation folded in: the app feeds [0,1] RGB
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed((image - self.mean) / self.std).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class TextEncoder(nn.Module):
    """Small bidirectional transformer over padded instruction tokens -> [b, L, fusion_dim]."""

    def __init__(self, c: StudentConfig):
        super().__init__()
        self.tok = nn.Embedding(c.vocab_size, c.text_dim)
        self.pos = nn.Parameter(torch.randn(1, c.lang_len, c.text_dim) * 0.02)
        self.blocks = nn.ModuleList(Block(c.text_dim, c.text_heads, c.mlp_ratio) for _ in range(c.text_depth))
        self.norm = nn.LayerNorm(c.text_dim, eps=1e-6)
        self.out = nn.Linear(c.text_dim, c.fusion_dim)

    def forward(self, lang_tokens: torch.Tensor, lang_mask: torch.Tensor) -> torch.Tensor:
        x = self.tok(lang_tokens) + self.pos
        bias = key_bias(lang_mask)
        for blk in self.blocks:
            x = blk(x, bias)
        return self.out(self.norm(x))


class StudentPolicy(nn.Module):
    """Per-frame part: image encoder + fusion + action head, given cached text features."""

    def __init__(self, c: StudentConfig):
        super().__init__()
        self.c = c
        self.image = ImageEncoder(c)
        self.img_proj = nn.Linear(c.vit_dim, c.fusion_dim)
        self.state_proj = nn.Linear(c.state_dim, c.fusion_dim)
        self.queries = nn.Parameter(torch.randn(1, c.chunk, c.fusion_dim) * 0.02)
        n_img = (c.image_size // c.patch) ** 2
        # type embedding per segment: image, text, state, action queries
        self.seg = nn.Parameter(torch.zeros(1, n_img + c.lang_len + 1 + c.chunk, c.fusion_dim))
        self.blocks = nn.ModuleList(Block(c.fusion_dim, c.fusion_heads, c.mlp_ratio) for _ in range(c.fusion_depth))
        self.norm = nn.LayerNorm(c.fusion_dim, eps=1e-6)
        self.head = nn.Linear(c.fusion_dim, c.action_dim)
        self.n_img = n_img

    def forward_with_text(self, image, text_feats, lang_mask, robot_state) -> torch.Tensor:
        b = image.shape[0]
        img = self.img_proj(self.image(image))
        st = self.state_proj(robot_state)[:, None, :]
        q = self.queries.expand(b, -1, -1)
        x = torch.cat([img, text_feats, st, q], dim=1) + self.seg
        ones_img = torch.ones(b, self.n_img, dtype=torch.float32, device=image.device)
        ones_tail = torch.ones(b, 1 + self.c.chunk, dtype=torch.float32, device=image.device)
        bias = key_bias(torch.cat([ones_img, lang_mask.to(torch.float32), ones_tail], dim=1))
        for blk in self.blocks:
            x = blk(x, bias)
        a = self.head(self.norm(x[:, -self.c.chunk:]))
        return torch.tanh(a)  # wheel commands in [-1, 1]; firmware scales to PWM


class StudentVLA(nn.Module):
    def __init__(self, c: StudentConfig | None = None):
        super().__init__()
        self.c = c or StudentConfig()
        self.text = TextEncoder(self.c)
        self.policy = StudentPolicy(self.c)
        self.apply(_init)

    def forward(self, image, lang_tokens, lang_mask, robot_state) -> torch.Tensor:
        return self.policy.forward_with_text(image, self.text(lang_tokens, lang_mask), lang_mask, robot_state)


def _init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


def param_counts(model: StudentVLA) -> dict[str, float]:
    def n(mod):
        return sum(p.numel() for p in mod.parameters()) / 1e6

    p = model.policy
    return {
        "total_M": round(n(model), 2),
        "image_encoder_M": round(n(p.image), 2),
        "text_encoder_M": round(n(model.text), 2),
        "text_embedding_M": round(n(model.text.tok), 2),
        "fusion_and_head_M": round(n(p) - n(p.image), 2),
    }
