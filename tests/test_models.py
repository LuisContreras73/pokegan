import torch

from pokegan import types as T
from pokegan.models import Discriminator, Generator


def _y(*names):
    return torch.from_numpy(T.encode(list(names)))[None]


def test_generator_shapes_range_and_params():
    torch.manual_seed(0)
    G = Generator(ch=8, attn=True)
    z = torch.randn(4, G.z_dim)
    y = torch.cat([_y("fire"), _y("water", "ice"), _y("ghost"), _y("fairy", "dragon")])
    x = G(z, y)
    assert x.shape == (4, 6, 96, 96)
    assert x.min() >= -1 and x.max() <= 1
    n = sum(p.numel() for p in Generator(ch=32).parameters())
    assert 6e6 < n < 14e6, n


def test_alpha_zero_gives_exact_white_background():
    G = Generator(ch=8, attn=False).eval()
    for head in G.heads:  # (no anular pesos: con SN, W=0 ⇒ σ=0 ⇒ 0/0 = NaN)
        with torch.no_grad():
            head.out.bias.copy_(torch.tensor([0.0, 0.0, 0.0, -200.0]))
    x = G(torch.randn(2, G.z_dim), torch.cat([_y("fire"), _y("grass")]))
    assert torch.equal(x, torch.ones_like(x))


def test_type_conditioning_and_transition_lambda():
    torch.manual_seed(0)
    G = Generator(ch=8, attn=False)
    G.train()
    G(torch.randn(16, G.z_dim), _y("fire").repeat(16, 1))  # poblar estadísticas BN
    G.eval()
    z = torch.randn(2, G.z_dim)
    a, b = G(z, _y("fire").repeat(2, 1)), G(z, _y("water").repeat(2, 1))
    assert (a - b).abs().max() > 1e-3                      # el tipo importa con λ=1
    a0, b0 = G(z, _y("fire").repeat(2, 1), lam=0.0), G(z, _y("water").repeat(2, 1), lam=0.0)
    assert torch.allclose(a0, b0)                          # con λ=0 G es incondicional


def test_discriminator_projection_is_linear_in_multi_hot():
    torch.manual_seed(0)
    D = Discriminator(ch=8, attn=True).eval()
    x = torch.rand(3, 6, 96, 96) * 2 - 1
    ya, yb = _y("fire").repeat(3, 1), _y("flying").repeat(3, 1)
    uc, pa = D(x, ya)
    _, pb = D(x, yb)
    uc2, pab = D(x, ya + yb)
    assert uc.shape == (3,) and pa.shape == (3,)
    assert torch.allclose(uc, uc2)
    assert torch.allclose(pab, pa + pb, atol=1e-5)
    n = sum(p.numel() for p in Discriminator(ch=32).parameters())
    assert 6e6 < n < 14e6, n
