"""Post-procesado de pixel art: determinista y SIN datos de entrenamiento.

Se muestra siempre al lado de la salida cruda de G (las métricas se calculan sobre la cruda).

Pasos para un par (frente, espalda):
  1. Fondo: relleno por inundación desde el borde sobre píxeles casi blancos
     (min(R,G,B) >= white_thr) -> blanco puro 255. Los blancos interiores (brillos, partes
     blancas del cuerpo) no tocan el borde y se conservan.
  2. Motas: se eliminan las componentes 8-conexas del primer plano con área < min_area.
  3. Paleta conjunta: el primer plano del lienzo [frente | espalda] (96x192) se cuantiza con
     median cut (<= colors colores, sin tramado), así ambas vistas comparten la misma paleta.
     El fondo no gasta colores: queda exactamente blanco.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

EIGHT_CONN = np.ones((3, 3), dtype=bool)


def background_mask(img: np.ndarray, white_thr: int = 235) -> np.ndarray:
    """True en el fondo: píxeles casi blancos conectados con el borde de la imagen.

    Se usa conexión 4 para que el relleno no se cuele por las diagonales del contorno.
    """
    near_white = img.min(axis=-1) >= white_thr
    lab, _ = ndimage.label(near_white)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, border[border > 0])


def remove_specks(fg: np.ndarray, min_area: int = 4, keep_frac: float = 0.0) -> np.ndarray:
    """Quita del primer plano las componentes 8-conexas con menos de max(min_area, keep_frac·área_mayor) píxeles.

    keep_frac > 0 elimina fragmentos flotantes pequeños relativos al cuerpo principal (p. ej. 0.05 = 5 %).
    """
    lab, n = ndimage.label(fg, structure=EIGHT_CONN)
    if n == 0:
        return fg
    areas = np.bincount(lab.ravel())
    thr = max(min_area, keep_frac * areas[1:].max()) if keep_frac > 0 else min_area
    small = areas < thr
    small[0] = False  # la etiqueta 0 es el fondo
    return fg & ~small[lab]


def clean_pair(front_u8: np.ndarray, back_u8: np.ndarray, colors: int = 15,
               white_thr: int = 235, min_area: int = 4, keep_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Limpia un par uint8 [96, 96, 3] y devuelve (frente, espalda) con la misma paleta."""
    # 1 y 2 se hacen por vista: cada vista tiene su propio borde y sus propias componentes.
    fg = np.concatenate(
        [remove_specks(~background_mask(v, white_thr), min_area, keep_frac) for v in (front_u8, back_u8)],
        axis=1,
    )
    canvas = np.concatenate([front_u8, back_u8], axis=1).astype(np.uint8)
    out = np.full_like(canvas, 255)
    if fg.any():
        # 3. Una sola "fila" con los píxeles del primer plano de ambas vistas -> paleta común.
        pixels = Image.fromarray(np.ascontiguousarray(canvas[fg].reshape(1, -1, 3)))
        q = pixels.quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                            dither=Image.Dither.NONE)
        out[fg] = np.asarray(q.convert("RGB")).reshape(-1, 3)
    w = front_u8.shape[1]
    return out[:, :w].copy(), out[:, w:].copy()


def clean_pairs(pairs: np.ndarray, **kw) -> np.ndarray:
    """Aplica clean_pair a un lote uint8 [n, 2, 96, 96, 3]."""
    return np.stack([np.stack(clean_pair(p[0], p[1], **kw)) for p in pairs])


def n_colors(img: np.ndarray) -> int:
    """Número de colores RGB distintos en una imagen (o en un par / lote)."""
    return len(np.unique(np.asarray(img).reshape(-1, 3), axis=0))
