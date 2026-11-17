import torch

from pokegan.augment import diff_augment, flip_pairs, rand_cutout, rand_translation


def _pair_batch(b=8):
    v = torch.rand(b, 3, 96, 96) * 2 - 1
    return torch.cat([v, v], 1)  # frente == espalda ⇒ si el aumento es pareado, siguen iguales


def test_translation_same_shift_both_views_and_white_fill():
    torch.manual_seed(0)
    x = _pair_batch()
    out = rand_translation(x, 0.125)
    assert torch.equal(out[:, :3], out[:, 3:])
    # una imagen negra desplazada debe mostrar blanco (+1) en el borde que entra
    black = -torch.ones(64, 6, 96, 96)
    o = rand_translation(black, 0.125)
    assert set(torch.unique(o).tolist()) <= {-1.0, 1.0} and (o == 1).any()


def test_cutout_same_region_both_views():
    torch.manual_seed(1)
    x = torch.ones(8, 6, 96, 96)
    o = rand_cutout(x, 0.5)
    assert torch.equal(o[:, :3], o[:, 3:])
    assert (o == 0).any() and torch.equal(o[:, 0], o[:, 5])


def test_diff_augment_keeps_pairs_and_gradient():
    torch.manual_seed(2)
    x = _pair_batch().requires_grad_(True)
    o = diff_augment(x)
    assert torch.equal(o[:, :3], o[:, 3:])
    o.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_flip_pairs_flips_both_views_together():
    torch.manual_seed(3)
    v = torch.arange(96.0).expand(16, 3, 96, 96)
    x = torch.cat([v, v], 1)
    o = flip_pairs(x, 0.5)
    assert torch.equal(o[:, :3], o[:, 3:])
    flipped = o[:, 0, 0, 0] == 95
    assert 0 < flipped.sum() < 16
