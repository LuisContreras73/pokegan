"""Figuras de resultados (textos en español, fondo blanco).

Todas las funciones guardan un archivo y devuelven su ruta. Los sprites se amplían con vecino
más cercano (nunca se suavizan) y se dibujan con un píxel de pantalla por píxel del lienzo.
Se usa matplotlib.figure.Figure directamente (sin pyplot): funciona igual en scripts y notebooks.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from . import types as T
from .postprocess import clean_pairs, n_colors
from .sample import cond_label, generate, interpolate, pair_canvas, upscale

# Combinaciones de tipos que NO aparecen en el set de entrenamiento del GAN.
UNSEEN = [
    ("normal", "rock"), ("normal", "bug"), ("normal", "steel"), ("normal", "ice"),
    ("poison", "ice"), ("ground", "fairy"), ("rock", "ghost"), ("bug", "dragon"),
    ("bug", "dark"), ("fire", "fairy"), ("dragon", "fairy"),
]
SEP_GRAY = 225  # color de las líneas que separan celdas
DPI = 100
POST_CAPTION = (
    "Post-procesado determinista (no usa datos de entrenamiento): fondo blanco por relleno desde "
    "el borde, eliminación de motas pequeñas y paleta conjunta frente+espalda de ≤{c} colores. "
    "Las métricas se calculan sobre la salida cruda de G."
)


def frequent_duals(label_bank, k: int = 12) -> list[tuple[str, ...]]:
    """Las k combinaciones de dos tipos más frecuentes en el banco de etiquetas [N, 18]."""
    y = np.asarray(label_bank) > 0.5
    duals = y[y.sum(1) == 2]
    keys, counts = np.unique(duals, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:k]
    return [tuple(T.decode(keys[i])) for i in order]


def save_grid(cells: np.ndarray, path, title=None, row_labels=None, col_labels=None,
              cell_labels=None, caption=None, scale: int = 2, pad: int = 4,
              col_rotation: float = 0) -> Path:
    """Dibuja una grilla de imágenes uint8 [R, C, h, w, 3] con etiquetas y la guarda en path."""
    R, C, h, w, _ = cells.shape
    canvas = np.full((R * (h + pad) + pad, C * (w + pad) + pad, 3), SEP_GRAY, np.uint8)
    for r in range(R):
        for c in range(C):
            y0, x0 = pad + r * (h + pad), pad + c * (w + pad)
            canvas[y0:y0 + h, x0:x0 + w] = cells[r, c]
    img = upscale(canvas, scale)

    fs = 6 + 4 * scale  # letra proporcional a la ampliación (legible al proyectar)
    fig = Figure(figsize=(img.shape[1] / DPI, img.shape[0] / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))  # el lienzo ocupa toda la figura: 1 píxel = 1 píxel
    ax.imshow(img, interpolation="nearest")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=fs)
    ax.xaxis.tick_top()
    centers_y = [(pad + r * (h + pad) + h / 2) * scale for r in range(R)]
    centers_x = [(pad + c * (w + pad) + w / 2) * scale for c in range(C)]
    ax.set_yticks(centers_y if row_labels else [], row_labels or [])
    ax.set_xticks(centers_x if col_labels else [], col_labels or [], rotation=col_rotation,
                  ha="left" if col_rotation else "center")
    if cell_labels is not None:
        for r in range(R):
            for c in range(C):
                if cell_labels[r][c]:
                    ax.text((pad + c * (w + pad) + 2) * scale, (pad + r * (h + pad) + 2) * scale,
                            cell_labels[r][c], fontsize=3 + 3 * scale, va="top", ha="left",
                            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    if title:
        ax.set_title(title, fontsize=fs + 4, pad=10)
    if caption:
        wrap = max(40, int(img.shape[1] / (0.8 * (fs - 2))))  # ~caracteres que caben a lo ancho
        fig.text(0.0, -0.01, textwrap.fill(caption, wrap), ha="left", va="top", fontsize=fs - 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    return path


def save_pairs_grid(pairs: np.ndarray, path, ncols: int = 4, labels=None, title=None,
                    caption=None, scale: int = 3) -> Path:
    """Grilla simple de pares [n, 2, 96, 96, 3] (cada celda = frente | espalda)."""
    n = len(pairs)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    h, w = pairs.shape[2:4]
    cells = np.full((nrows, ncols, h, 2 * w, 3), 255, np.uint8)
    cell_labels = [[None] * ncols for _ in range(nrows)]
    for i, p in enumerate(pairs):
        r, c = divmod(i, ncols)
        cells[r, c] = pair_canvas(p)
        cell_labels[r][c] = labels[i] if labels else None
    return save_grid(cells, path, title=title, cell_labels=cell_labels, caption=caption,
                     scale=scale)


def grid_by_type(G, conditions, n_z: int, seed: int, path, post: bool = False, lam: float = 1.0,
                 truncation: float | None = None, colors: int = 15, scale: int = 2,
                 title: str | None = None) -> Path:
    """Filas = condiciones (1-2 tipos), columnas = n_z valores de z (los mismos en todas las filas).

    Cada celda muestra frente | espalda generados desde un único z.
    """
    rows = []
    for cond in conditions:
        pairs = generate(G, cond, n_z, seed=seed, truncation=truncation, lam=lam)
        if post:
            pairs = clean_pairs(pairs, colors=colors)
        rows.append([pair_canvas(p) for p in pairs])
    title = title or "Generación por tipo (filas) y z (columnas) · cada celda: frente | espalda"
    return save_grid(
        np.array(rows), path, title=title,
        row_labels=[cond_label(c) for c in conditions],
        col_labels=[f"z{j + 1}" for j in range(n_z)],
        caption=POST_CAPTION.format(c=colors) if post else None, scale=scale,
    )


def same_z_all_types(G, n_z: int, seed: int, path, views: str = "front", lam: float = 1.0,
                     truncation: float | None = None, scale: int = 2) -> Path:
    """Filas = z fijos, columnas = los 18 monotipos.

    Muestra la división de trabajo: z decide el individuo (silueta, pose), y decide el tipo
    (paleta, rasgos). views='front' muestra solo el frente (más legible); 'both' frente | espalda.
    """
    cols = []
    for t in T.TYPES:
        pairs = generate(G, [t], n_z, seed=seed, truncation=truncation, lam=lam)
        cols.append([pair_canvas(p) if views == "both" else p[0] for p in pairs])
    cells = np.array(cols).transpose(1, 0, 2, 3, 4)  # [n_z, 18, h, w, 3]
    vista = "frente | espalda" if views == "both" else "vista frontal"
    return save_grid(
        cells, path, title=f"Mismo z (filas), distinto tipo (columnas) · {vista}",
        row_labels=[f"z{j + 1}" for j in range(n_z)],
        col_labels=[T.TYPES_ES[t] for t in T.TYPES], col_rotation=45, scale=scale,
    )


def _interp_title(types_a, types_b, seed_a, seed_b) -> str:
    la, lb = cond_label(types_a), cond_label(types_b)
    if la == lb:
        return f"Interpolación en z con tipo fijo ({la}): semilla {seed_a} → {seed_b}"
    if seed_a == seed_b:
        return f"Interpolación de tipo {la} → {lb} con z fijo (semilla {seed_a})"
    return f"Interpolación {la} (semilla {seed_a}) → {lb} (semilla {seed_b})"


def interpolation_strip(G, types_a, types_b, seed_a: int, seed_b: int, path, steps: int = 8,
                        lam: float = 1.0, truncation: float | None = None,
                        scale: int = 2) -> Path:
    """Tira de interpolación: fila superior = frente, inferior = espalda, columnas = t de 0 a 1."""
    frames = interpolate(G, types_a, types_b, seed_a, seed_b, steps, lam, truncation)
    ts = np.linspace(0, 1, steps)
    return save_grid(
        frames.transpose(1, 0, 2, 3, 4), path, title=_interp_title(types_a, types_b, seed_a, seed_b),
        row_labels=["Frente", "Espalda"], col_labels=[f"t={t:.2f}" for t in ts], scale=scale,
    )


def interpolation_gif(G, types_a, types_b, seed_a: int, seed_b: int, path, steps: int = 20,
                      lam: float = 1.0, truncation: float | None = None, scale: int = 3,
                      duration_ms: int = 90, pingpong: bool = True) -> Path:
    """GIF animado de la interpolación (frente | espalda) con una barra de texto arriba."""
    frames = interpolate(G, types_a, types_b, seed_a, seed_b, steps, lam, truncation)
    la, lb = cond_label(types_a), cond_label(types_b)
    # DejaVu Sans (incluida en matplotlib) tiene tildes; la fuente por defecto de PIL no (Drag□n).
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 16)
    images = []
    for t, pair in zip(np.linspace(0, 1, steps), frames):
        if la != lb:
            text = f"{la} {100 * (1 - t):.0f}% + {lb} {100 * t:.0f}%"
        else:
            text = f"{la} | z: semilla {seed_a} -> {seed_b} | t={t:.2f}"
        sprite = Image.fromarray(upscale(pair_canvas(pair), scale))
        im = Image.new("RGB", (sprite.width, sprite.height + 28), "white")
        im.paste(sprite, (0, 28))
        ImageDraw.Draw(im).text((6, 5), text, fill=(0, 0, 0), font=font)
        images.append(im)
    if pingpong:
        images += images[-2:0:-1]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return path


def raw_vs_post(G, conditions, seed: int, path, n_z: int = 2, colors: int = 15, lam: float = 1.0,
                truncation: float | None = None, scale: int = 2) -> Path:
    """Salida cruda de G al lado de su versión post-procesada, con el número de colores de cada una."""
    rows, counts = [], []
    for cond in conditions:
        raw = generate(G, cond, n_z, seed=seed, truncation=truncation, lam=lam)
        post = clean_pairs(raw, colors=colors)
        row, cnt = [], []
        for r, p in zip(raw, post):
            row += [pair_canvas(r), pair_canvas(p)]
            cnt += [f"{n_colors(r)} colores", f"{n_colors(p)} colores"]
        rows.append(row)
        counts.append(cnt)
    col_labels = [f"z{j + 1} {kind}" for j in range(n_z) for kind in ("crudo", "post")]
    return save_grid(
        np.array(rows), path, title="Salida cruda de G vs. post-procesada",
        row_labels=[cond_label(c) for c in conditions], col_labels=col_labels,
        cell_labels=counts, caption=POST_CAPTION.format(c=colors), scale=scale,
    )


# (título, columnas del log, nombres en la leyenda)
CURVE_PANELS = [
    ("Pérdidas (hinge)", ["d_loss", "g_loss"], ["D", "G"]),
    ("Salida media de D", ["d_real", "d_fake", "d_mis"], ["reales", "falsos", "desajustados"]),
    ("Sobreajuste de D: E[signo D(real)] (alerta > 0.7)", ["r_t", "r_v"],
     ["r_t (entrenamiento)", "r_v (validación)"]),
    ("λ_t: transición incondicional → condicional", ["lam"], ["λ_t"]),
    ("Diversidad: distancia media entre muestras del mismo tipo (↑)", ["div"], ["div"]),
    ("Penalización LeCam", ["lecam"], ["LeCam"]),
    ("KID (↓ mejor)", ["kid_front", "kid_back"], ["frente", "espalda"]),
    ("mAP de tipo (↑ mejor)", ["type_map"], ["mAP"]),
    ("Coherencia frente/espalda (↑ mejor)", ["coherence"], ["coherencia"]),
]


def training_curves(log_csv, path) -> Path:
    """Curvas de entrenamiento desde log.csv (las columnas opcionales se omiten si no hay datos)."""
    import pandas as pd

    df = pd.read_csv(log_csv)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["step"])
    # Reanudaciones: un paso repetido conserva la última fila escrita (orden estable: si no,
    # quicksort mezcla los duplicados y "last" puede quedarse con la fila previa al corte).
    df = df.sort_values("step", kind="stable").drop_duplicates("step", keep="last")
    panels = [(t, [(c, n) for c, n in zip(cols, names) if c in df and df[c].notna().any()])
              for t, cols, names in CURVE_PANELS]
    panels = [(t, s) for t, s in panels if s]
    ncols = 3
    nrows = max(1, -(-len(panels) // ncols))
    fig = Figure(figsize=(5 * ncols, 3.3 * nrows), dpi=DPI, facecolor="white", layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax, (title, series) in zip(axes, panels):
        for col, name in series:
            s = df[["step", col]].dropna()
            if len(s) > 30:  # serie ruidosa: cruda tenue + media móvil
                line, = ax.plot(s.step, s[col], alpha=0.25, lw=0.8)
                win = max(3, len(s) // 50)
                ax.plot(s.step, s[col].rolling(win, min_periods=1, center=True).mean(),
                        color=line.get_color(), lw=1.6, label=name)
            else:  # métrica esporádica (KID, mAP...): puntos unidos
                ax.plot(s.step, s[col], "o-", ms=3, label=name)
        if any(c in ("r_t", "r_v") for c, _ in series):
            ax.axhline(0.7, color="gray", ls="--", lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("iteración")
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=9)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Curvas de entrenamiento", fontsize=14)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    return path
