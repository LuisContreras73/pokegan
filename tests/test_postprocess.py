import numpy as np

from pokegan.postprocess import background_mask, clean_pair, clean_pairs, n_colors, remove_specks


def _sprite(seed: int) -> np.ndarray:
    """Sprite sintético: fondo casi blanco con ruido, cuerpo con degradado (muchos colores),
    un hueco blanco interior y una mota aislada de 1 píxel."""
    rng = np.random.default_rng(seed)
    img = rng.integers(238, 256, size=(96, 96, 3)).astype(np.uint8)  # fondo "sucio"
    yy, xx = np.mgrid[:96, :96]
    body = (yy - 48) ** 2 + (xx - 48) ** 2 < 30 ** 2
    grad = np.stack([xx * 2, yy * 2, 255 - xx - yy // 2], -1).clip(0, 230).astype(np.uint8)
    img[body] = grad[body] + rng.integers(0, 20, size=(int(body.sum()), 3)).astype(np.uint8)
    img[45:50, 45:50] = 250  # blanco interior (rodeado por el cuerpo): debe conservarse
    img[5, 5] = (10, 10, 10)  # mota aislada
    return img


def test_background_mask_and_specks():
    img = _sprite(0)
    bg = background_mask(img)
    assert bg[0, 0] and bg[95, 95] and not bg[48, 48] and not bg[47, 47]
    fg = remove_specks(~bg, min_area=4)
    assert not fg[5, 5] and fg[48, 20]


def test_clean_pair_background_palette_and_determinism():
    front, back = _sprite(0), _sprite(1)[:, ::-1].copy()
    assert n_colors(np.stack([front, back])) > 100
    f, b = clean_pair(front, back, colors=15)
    assert f.shape == front.shape and f.dtype == np.uint8 and b.shape == back.shape
    for v in (f, b):
        for y, x in [(0, 0), (0, 95), (95, 0), (95, 95)]:
            assert (v[y, x] == 255).all()
        assert (v[5, 5] == 255).all()  # mota eliminada (en el frente; en la espalda es fondo)
    # todo el fondo conectado al borde queda exactamente blanco
    assert (f[background_mask(front)] == 255).all()
    # paleta conjunta: <= colors + blanco en la unión de ambas vistas
    assert n_colors(np.concatenate([f, b], axis=1)) <= 16
    f2, b2 = clean_pair(front, back, colors=15)
    assert (f == f2).all() and (b == b2).all()


def test_clean_pair_all_white_and_batch():
    white = np.full((96, 96, 3), 255, np.uint8)
    f, b = clean_pair(white, white)
    assert (f == 255).all() and (b == 255).all()
    pairs = np.stack([np.stack([_sprite(0), _sprite(1)])] * 2)
    out = clean_pairs(pairs, colors=8)
    assert out.shape == pairs.shape and n_colors(out[0]) <= 9
