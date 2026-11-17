from pathlib import Path

import numpy as np
import pytest

from pokegan import types as T
from pokegan.data import (
    combo_keys, load_cache, load_curated, sampling_weights, species_split, to_tensor_pairs,
)

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "pairs96.npz"
needs_cache = pytest.mark.skipif(not CACHE.exists(), reason="ejecuta scripts/prepare_data.py primero")


def test_encode_decode_roundtrip_es_en():
    y = T.encode(["Fuego", "flying"])
    assert y.sum() == 2 and T.decode(y) == ["flying", "fire"]
    assert T.label(y) == "Volador/Fuego"
    assert T.normalize_type("electrico") == T.normalize_type("Eléctrico") == "electric"
    assert T.normalize_type("PSIQUICO") == "psychic" and T.normalize_type("dragon") == "dragon"
    with pytest.raises(ValueError):
        T.encode(["fire", "fire"])
    with pytest.raises(ValueError):
        T.encode(["pikachu"])


def test_all_conditions_is_171():
    assert len(T.all_conditions()) == 171


def test_species_split_is_disjoint_and_deterministic():
    sp = np.repeat(np.arange(200), 5)
    a, b = species_split(sp, 0.2, seed=1), species_split(sp, 0.2, seed=1)
    assert (a == b).all()
    assert not set(sp[a]) & set(sp[~a])
    assert 0.1 < a.mean() < 0.3


@needs_cache
def test_cache_shapes_and_labels():
    d = load_cache(CACHE, drop_gray=False)
    assert len(d) == 12806
    assert d.front.shape == (12806, 96, 96, 3) and d.front.dtype == np.uint8
    assert d.back.shape == d.front.shape
    k = d.y.sum(1)
    assert set(np.unique(k)) == {1.0, 2.0}
    # fondo blanco en la esquina de todas las imágenes
    assert (d.front[:, 0, 0] == 255).all() and (d.back[:, 0, 0] == 255).all()


@needs_cache
def test_gray_filter_and_weights():
    d = load_cache(CACHE, drop_gray=True)
    assert len(d) == 12806 - 987
    w = sampling_weights(d)
    assert np.isclose(w.sum(), 1.0) and (w > 0).all()
    # rebalanceo: la combinación más rara debe ganar probabilidad relativa vs uniforme
    keys = combo_keys(d.y)
    u, c = np.unique(keys, return_counts=True)
    rare = keys == u[c.argmin()]
    assert w[rare].sum() > rare.mean()


@needs_cache
def test_curated_counts():
    tr, va = load_curated(CACHE, val_frac=0.05, seed=0)
    assert len(tr) + len(va) == 8085
    assert not set(tr.species_id) & set(va.species_id)
    assert 286 <= len(va) <= 406
    names = tr.collection_names[np.unique(tr.collection)]
    assert not any(str(n).startswith(("generation-1", "generation-2")) for n in names)
    x = to_tensor_pairs(va)
    assert tuple(x.shape) == (len(va), 6, 96, 96) and str(x.dtype) == "torch.uint8"
    assert (x[:, :3].numpy() == va.front.transpose(0, 3, 1, 2)).all()


@needs_cache
def test_weights_shiny_factor():
    tr, _ = load_curated(CACHE)
    w = sampling_weights(tr, species_power=0.0, combo_power=0.0, shiny_factor=0.5)
    shiny, normal = w[tr.is_shiny == 1], w[tr.is_shiny == 0]
    assert np.allclose(shiny / normal[0], 0.5)
