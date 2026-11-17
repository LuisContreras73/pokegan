"""Figura resumen: Pokémon REALES vs GENERADOS por tipo (frente | espalda), lado a lado.

Filas = condiciones de tipo; columnas izquierdas = sprites reales de ese tipo (especies distintas, no shiny);
columnas derechas = criaturas generadas por el modelo final con esa misma condición (z distintos, sin escoger a mano).

Uso:
    python scripts/make_summary.py --ckpt weights/pokegan_pura.pt --out results/RESUMEN_real_vs_generado.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import types as T  # noqa: E402
from pokegan.data import load_curated  # noqa: E402
from pokegan.sample import generate, load_generator  # noqa: E402

CONDS = [("fire",), ("water",), ("grass",), ("electric",), ("ghost",), ("ice",), ("dragon",), ("fire", "flying"),
         ("bug", "poison")]


def real_examples(d, cond, k, rng):
    y = T.encode(list(cond))
    match = np.where((np.abs(d.y - y).sum(1) == 0) & (d.is_shiny == 0))[0]
    out, seen = [], set()
    for i in rng.permutation(match):
        if d.species_id[i] in seen:
            continue
        seen.add(d.species_id[i])
        out.append(np.stack([d.front[i], d.back[i]]))
        if len(out) == k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "weights" / "pokegan_pura.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "RESUMEN_real_vs_generado.png"))
    ap.add_argument("--k-real", type=int, default=3)
    ap.add_argument("--k-gen", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G, info = load_generator(a.ckpt, ema=True)
    tr, _ = load_curated(ROOT / "data" / "cache" / "pairs96.npz")
    rng = np.random.default_rng(a.seed)
    nr, ng = a.k_real, a.k_gen
    fig, axes = plt.subplots(len(CONDS), nr + ng, figsize=(2.3 * (nr + ng) + 1.2, 1.25 * len(CONDS) + 0.9),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    for r, cond in enumerate(CONDS):
        reals = real_examples(tr, cond, nr, rng)
        gens = generate(G, list(cond), ng, seed=a.seed + 10 * r, lam=info["lam"])  # [ng, 2, 96, 96, 3]
        cells = [(p, "real") for p in reals] + [(g, "gen") for g in gens]
        for c in range(nr + ng):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(cells):
                pair, kind = cells[c]
                ax.imshow(np.concatenate([pair[0], pair[1]], axis=1), interpolation="nearest")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#2b7bb9" if kind == "real" else "#d9534f")
                    sp.set_linewidth(2.2)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(T.label(T.encode(list(cond))), rotation=0, ha="right", va="center", fontsize=12)
    for c in range(nr + ng):
        axes[0, c].set_title("REAL" if c < nr else "GENERADO", fontsize=12,
                             color="#2b7bb9" if c < nr else "#d9534f", fontweight="bold")
    fig.suptitle(f"Reales (azul) vs generados por PokeGAN (rojo) · cada celda: frente | espalda · "
                 f"generados sin escoger a mano (semillas fijas) · paso {info['step']}", fontsize=13)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"Guardado en {a.out}")


if __name__ == "__main__":
    main()
