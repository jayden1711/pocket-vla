"""Quick checks for the student model (plain python or pytest):

    training/.venv/bin/python training/tests/test_student.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from student import StudentConfig, StudentVLA, param_counts  # noqa: E402


def _inputs(c, n_valid=6):
    mask = torch.zeros(1, c.lang_len, dtype=torch.int32)
    mask[:, :n_valid] = 1
    return (torch.rand(1, 3, c.image_size, c.image_size), torch.randint(0, c.vocab_size, (1, c.lang_len)),
            mask, torch.zeros(1, c.state_dim))


def test_shape_range_and_size():
    torch.manual_seed(0)
    c = StudentConfig()
    m = StudentVLA(c).eval()
    with torch.no_grad():
        a = m(*_inputs(c))
    assert a.shape == (1, c.chunk, c.action_dim)
    assert a.abs().max() <= 1.0
    assert 70 <= param_counts(m)["total_M"] <= 82


def test_padding_tokens_are_ignored():
    torch.manual_seed(0)
    c = StudentConfig()
    m = StudentVLA(c).eval()
    img, tok, mask, st = _inputs(c)
    tok2 = tok.clone()
    tok2[:, 6:] = 0  # change only padded positions
    with torch.no_grad():
        assert torch.allclose(m(img, tok, mask, st), m(img, tok2, mask, st), atol=1e-5)


def test_split_equals_full():
    torch.manual_seed(0)
    c = StudentConfig()
    m = StudentVLA(c).eval()
    img, tok, mask, st = _inputs(c)
    with torch.no_grad():
        full = m(img, tok, mask, st)
        split = m.policy.forward_with_text(img, m.text(tok, mask), mask, st)
    assert torch.allclose(full, split)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
