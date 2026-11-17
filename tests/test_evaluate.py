"""Tests de evaluación (CPU, datos sintéticos)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pokegan import evaluate as E
from pokegan import types as T

INCEPTION_W = Path.home() / ".cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
needs_inception = pytest.mark.skipif(not INCEPTION_W.exists(), reason="pesos Inception no en caché")


def sprite(color, size=96, box=(30, 70)):
    """Sprite sintético: fondo blanco con un cuadrado de un color."""
    img = np.full((size, size, 3), 255, np.uint8)
    img[box[0]:box[1], box[0]:box[1]] = color
    return img


# ---------------------------------------------------------------- paleta

def test_palette_hi_identical_is_one():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    assert E.palette_hi(x, x) == pytest.approx(1.0)


def test_palette_hi_disjoint_is_zero():
    assert E.palette_hi(sprite((200, 0, 0)), sprite((0, 0, 200))) == 0.0


def test_palette_hi_ignores_white_and_handles_empty():
    white = np.full((96, 96, 3), 255, np.uint8)
    # el fondo blanco no cuenta: mismo color en distinta posición/tamaño -> 1
    assert E.palette_hi(sprite((10, 120, 10)), sprite((10, 120, 10), box=(5, 20))) == pytest.approx(1.0)
    # una vista sin primer plano -> 0 (documentado: no hay paleta compartida)
    assert E.palette_hi(white, sprite((10, 120, 10))) == 0.0
    assert E.palette_hi(white, white) == 0.0


def test_palette_hi_batch_matches_single():
    rng = np.random.default_rng(1)
    f = rng.integers(0, 256, (5, 96, 96, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (5, 96, 96, 3), dtype=np.uint8)
    batch = E.palette_hi_batch(f, b, chunk=2)
    single = [E.palette_hi(f[i], b[i]) for i in range(5)]
    np.testing.assert_allclose(batch, single)


# ---------------------------------------------------------------- tipo

def test_type_metrics_perfect():
    rng = np.random.default_rng(0)
    conds = T.all_conditions()
    y = np.stack([T.encode(conds[i]) for i in rng.integers(0, len(conds), 200)])
    m = E.type_metrics(10 * (2 * y - 1), y)
    assert m["mAP"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["set_acc"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in m["per_type_recall"].values())


def test_type_metrics_set_acc_hand_example():
    y = np.stack([T.encode(["fuego", "volador"]), T.encode(["agua"]), T.encode(["planta", "veneno"])])
    p = np.full((3, 18), 0.1, np.float32)
    p[0, [T.TYPE_TO_IDX["fire"], T.TYPE_TO_IDX["flying"]]] = [0.9, 0.8]   # acierta el par
    p[1, T.TYPE_TO_IDX["water"]] = 0.7                                      # acierta el monotipo
    p[2, [T.TYPE_TO_IDX["grass"], T.TYPE_TO_IDX["bug"]]] = [0.9, 0.8]       # top-2 = planta/bicho
    p[2, T.TYPE_TO_IDX["poison"]] = 0.5
    assert E.type_metrics(p, y)["set_acc"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- vecinos y diversidad

def test_nearest_neighbors_finds_exact_copy():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 256, (50, 96, 96, 3), dtype=np.uint8)
    gen = train[[7, 31]].copy()
    idx, dist = E.nearest_neighbors(gen, train, k=3, device="cpu", gen_chunk=1, train_chunk=16)
    assert idx[:, 0].tolist() == [7, 31]
    assert dist[:, 0] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert (dist[:, 1] > 1.0).all() and (np.diff(dist, axis=1) >= 0).all()


def test_nn_figure_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    train = rng.integers(0, 256, (10, 96, 96, 3), dtype=np.uint8)
    idx, dist = E.nearest_neighbors(train[:2], train, k=3, device="cpu")
    p = E.nn_figure(train[:2], train, idx, tmp_path / "nn.png", dist=dist)
    assert p.exists() and p.stat().st_size > 0


def test_diversity_helpers():
    same = torch.ones(10, 8)
    assert E.mean_pairwise_cosine_distance(same) == pytest.approx(0.0, abs=1e-9)
    assert E.vendi_score(same) == pytest.approx(1.0)
    assert E.vendi_score(torch.eye(6)) == pytest.approx(6.0)
    imgs = np.zeros((3, 4, 4, 3), np.uint8)
    imgs[1] = 255
    assert E.mean_pairwise_l1(imgs) == pytest.approx(2 / 3)   # pares (0,1),(1,2)=1 y (0,2)=0
    d = E.diversity(torch.randn(20, 8), groups=np.repeat([0, 1], 10), imgs_u8=np.zeros((20, 4, 4, 3)))
    assert set(d) == {"cosine", "vendi", "l1"}


# ---------------------------------------------------------------- KID / FID

def test_kid_matches_torch_fidelity_and_identical_is_zero():
    from torch_fidelity.metric_kid import kernel_mmd

    g = torch.Generator().manual_seed(0)
    a = torch.rand(40, 32, generator=g, dtype=torch.float64)
    b = torch.rand(40, 32, generator=g, dtype=torch.float64) + 0.3
    # con subset_size = n cada subconjunto es todo el conjunto -> comparación exacta
    mine, _ = E.kid_from_feats(a, b, subset_size=40, subsets=3)
    ref = kernel_mmd(a.numpy(), b.numpy(), kid_kernel="poly", kid_kernel_poly_degree=3,
                     kid_kernel_poly_gamma=None, kid_kernel_poly_coef0=1)
    assert mine == pytest.approx(float(ref), rel=1e-9)
    same, _ = E.kid_from_feats(a, a.clone(), subset_size=20, subsets=20)
    assert abs(same) < 0.05 * mine


def test_fid_from_feats_matches_scipy():
    from scipy import linalg

    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 6))
    b = rng.normal(loc=0.5, scale=1.5, size=(200, 6))
    s_a, s_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ref = ((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s_a + s_b - 2 * linalg.sqrtm(s_a @ s_b).real)
    assert E.fid_from_feats(torch.from_numpy(a), torch.from_numpy(b)) == pytest.approx(ref, rel=1e-6)
    assert E.fid_from_feats(torch.from_numpy(a), torch.from_numpy(a)) == pytest.approx(0.0, abs=1e-6)


@needs_inception
def test_kid_fid_inception_identical_sets_tiny():
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 200, (6, 3))
    x = torch.from_numpy(np.stack([sprite(c) for c in colors]).transpose(0, 3, 1, 2).copy())
    noise = torch.from_numpy(rng.integers(0, 256, x.shape, dtype=np.uint8))
    diff = E.kid_fid(x, noise, device="cpu", fid=True, subset_size=4, subsets=20)
    assert diff["kid"] > 0 and np.isfinite(diff["fid"])
    f = E.inception_features(x, "cpu")
    same, _ = E.kid_from_feats(f, f.clone(), subset_size=4, subsets=20)
    assert abs(same) < 0.1 * diff["kid"]


# ---------------------------------------------------------------- clasificador + quick_eval

class DummyG(torch.nn.Module):
    """G de juguete con la interfaz de Generator: G(z, y, lam) -> [B, 6, 96, 96] en [-1, 1]."""

    z_dim = 8

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8 + 18, 6)
        self.bn = torch.nn.BatchNorm1d(6)

    def forward(self, z, y, lam=1.0):
        h = torch.tanh(self.bn(self.lin(torch.cat([z, lam * y], 1))))
        return h[:, :, None, None].expand(-1, -1, 96, 96).contiguous()


def tiny_pairdata(n, seed):
    from pokegan.data import PairData

    rng = np.random.default_rng(seed)
    conds = T.all_conditions()
    y = np.stack([T.encode(conds[i]) for i in rng.integers(0, 18, n)])
    colors = rng.integers(0, 200, (n, 3))
    front = np.stack([sprite(c) for c in colors])
    back = np.stack([sprite(c, box=(20, 60)) for c in colors])
    return PairData(front=front, back=back, y=y, species_id=np.arange(n) + seed * 1000,
                    is_shiny=np.zeros(n, np.uint8), collection=np.zeros(n, np.int16),
                    collection_names=np.array(["x"]), species_name=np.array(["s"] * n),
                    saturation=np.ones((n, 2), np.float32))


def test_coherence_reference_on_synthetic():
    d = tiny_pairdata(30, 0)
    ref = E.coherence_reference(d, n=30)
    assert ref["matched"] == pytest.approx(1.0) and ref["random"] < 0.5


@needs_inception
def test_train_classifier_and_quick_eval(tmp_path):
    from pokegan.type_classifier import load_classifier, train_type_classifier

    train, val = tiny_pairdata(12, 0), tiny_pairdata(6, 1)
    m = train_type_classifier(train, val, tmp_path / "clf", epochs=1, batch=8, device="cpu")
    assert {"mAP", "f1_macro", "set_acc"} <= set(m)
    clf, th = load_classifier(tmp_path / "clf" / "best.pt", "cpu")
    assert th.shape == (18,) and not clf.training

    torch.save({"front": torch.rand(16, 2048, dtype=torch.float64),
                "back": torch.rand(16, 2048, dtype=torch.float64)}, tmp_path / "feats.pt")
    G = DummyG().train()
    bank = torch.from_numpy(train.y).to(torch.uint8)
    out = E.quick_eval(G, 1.0, bank, device="cpu", n=3, real_feats_path=tmp_path / "feats.pt",
                       clf_path=tmp_path / "clf" / "best.pt")
    assert set(out) == {"kid_front", "kid_back", "type_map", "coherence"}
    assert all(np.isfinite(v) for v in out.values())
    assert G.training  # restaura el modo original

    out = E.quick_eval(G, 1.0, bank, device="cpu", n=3, real_feats_path=tmp_path / "feats.pt",
                       clf_path=tmp_path / "missing.pt")
    assert "type_map" not in out


def test_classifier_augment_pads_white_and_keeps_colors():
    from pokegan.type_classifier import random_shift_flip

    torch.manual_seed(0)
    x = -torch.ones(32, 3, 16, 16)                    # imágenes negras
    out = random_shift_flip(x, max_shift=4)
    assert out.shape == x.shape
    assert set(out.unique().tolist()) <= {-1.0, 1.0}  # solo negro original o relleno blanco (+1)
    assert (out == 1).any() and (out == -1).any()
