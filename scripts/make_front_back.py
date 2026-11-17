"""Figura explícita del requisito central: UN MISMO z → FRENTE y ESPALDA de la MISMA criatura.

Cada columna es una criatura: arriba su vista FRONTAL, abajo su vista TRASERA (ambas salen de un único z y del tipo).
La primera columna de cada tipo es un par REAL de referencia.

Uso:
    python scripts/make_front_back.py --ckpt weights/pokegan_pura.pt --out results/FRENTE_ESPALDA.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import types as T  # noqa: E402
from pokegan.data import load_curated  # noqa: E402
from pokegan.sample import generate, load_generator  # noqa: E402

CONDS = [("fire",), ("water",), ("grass",), ("electric",), ("ghost",), ("fire", "flying")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "weights" / "pokegan_pura.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "FRENTE_ESPALDA.png"))
    ap.add_argument("--n", type=int, default=5, help="criaturas generadas por tipo")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--post", action="store_true", help="post-proceso pixel art declarado: paleta conjunta, fondo limpio, sin fragmentos")
    ap.add_argument("--colors", type=int, default=10)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G, info = load_generator(a.ckpt, ema=True)
    tr, _ = load_curated(ROOT / "data" / "cache" / "pairs96.npz")
    rng = np.random.default_rng(a.seed)
    ncol = 1 + a.n
    nrow = 2 * len(CONDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.75 * ncol + 1.6, 1.75 * nrow + 0.8),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.05})
    for r, cond in enumerate(CONDS):
        y = T.encode(list(cond))
        idx = np.where((np.abs(tr.y - y).sum(1) == 0) & (tr.is_shiny == 0))[0]
        i = rng.choice(idx)
        gens = generate(G, list(cond), a.n, seed=a.seed + 100 * r, lam=info["lam"])  # [n, 2, 96, 96, 3]
        if a.post:
            from pokegan.postprocess import clean_pair
            gens = [np.stack(clean_pair(g[0], g[1], colors=a.colors, keep_frac=0.15)) for g in gens]
        cols = [(tr.front[i], tr.back[i], "real")] + [(g[0], g[1], "gen") for g in gens]
        for c, (f, b, kind) in enumerate(cols):
            color = "#2b7bb9" if kind == "real" else "#d9534f"
            for k, img in enumerate((f, b)):
                ax = axes[2 * r + k, c]
                ax.imshow(img, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(color)
                    sp.set_linewidth(2.0)
            if r == 0:
                axes[0, c].set_title("REAL\n(referencia)" if kind == "real" else f"GENERADO\nz #{c}", fontsize=11,
                                     color=color, fontweight="bold")
        axes[2 * r, 0].set_ylabel(f"{T.label(y)}\nFRENTE", rotation=0, ha="right", va="center", fontsize=11)
        axes[2 * r + 1, 0].set_ylabel("ESPALDA", rotation=0, ha="right", va="center", fontsize=11, color="#555555")
    fig.suptitle("Un mismo z → vista FRONTAL (arriba) y TRASERA (abajo) de la MISMA criatura · condición = solo el tipo\n"
                 f"Azul: par real de referencia · Rojo: generados por PokeGAN sin escoger a mano · paso {info['step']}"
                 + (f" · POST-PROCESO declarado: paleta conjunta de {a.colors} colores + fondo limpio + sin fragmentos" if a.post else ""),
                 fontsize=13, y=0.995)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"Guardado en {a.out}")


if __name__ == "__main__":
    main()
