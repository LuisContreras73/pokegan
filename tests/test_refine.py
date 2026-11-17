import torch

from pokegan import refine as R
from pokegan import types as T
from pokegan.models import Discriminator, Generator


def _models():
    torch.manual_seed(0)
    G = Generator(ch=8, attn=False).eval()
    D = Discriminator(ch=8, attn=False).eval()
    return G, D


def test_chunk_mask_and_coarse_truncation_only_touch_chunks():
    m = R.chunk_mask(120, (0, 1, 2))
    assert m[:60].eq(1).all() and m[60:].eq(0).all()
    z = torch.randn(64, 120) * 3
    zt = R.truncate(z, 0.7, chunks=(0, 1, 2), generator=torch.Generator().manual_seed(0))
    assert zt[:, :60].abs().max() <= 0.7
    assert torch.equal(zt[:, 60:], z[:, 60:])          # el detalle (z3–z5) no cambia
    zg = R.truncate(z, 0.5)
    assert zg.abs().max() <= 0.5


def test_d_logit_guidance_is_linear_in_w():
    G, D = _models()
    y = torch.from_numpy(T.encode(["fire", "flying"]))[None].repeat(4, 1)
    x = G(torch.randn(4, 120), y)
    uc, pj = D(x, y)
    assert torch.allclose(R.d_logit(D, x, y, n_aug=0, w=2.0), uc + 2 * pj, atol=1e-5)


def test_count_fragments():
    x = torch.full((2, 3, 96, 96), 255, dtype=torch.uint8)
    x[0, :, 10:20, 10:20] = 0                         # un cuerpo
    x[1, :, 10:20, 10:20] = 0
    x[1, :, 50:60, 50:60] = 0                         # dos cuerpos
    x[1, :, 80, 80] = 0                               # mota de 1 px (no cuenta)
    assert R.count_fragments(x).tolist() == [1, 2]


def test_sir_and_dgflow_shapes_and_coarse_mask():
    G, D = _models()
    y = torch.from_numpy(T.encode(["water"]))[None].repeat(3, 1)
    z = R.sir(G, D, y, k=4, n_aug=1, generator=torch.Generator().manual_seed(0), w=2.0, trunc_coarse=0.7)
    assert z.shape == (3, 120) and z[:, :60].abs().max() <= 0.7
    z0 = torch.randn(3, 120)
    z1 = R.dgflow(G, D, z0, y, steps=2, n_aug=1, chunks=(0, 1, 2))
    assert z1.shape == z0.shape
    assert torch.equal(z1[:, 60:], z0[:, 60:])        # solo se movieron z0–z2
    assert not torch.equal(z1[:, :60], z0[:, :60])
