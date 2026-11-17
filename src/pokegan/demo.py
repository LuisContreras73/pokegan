"""Demo simple para Colab / Jupyter: cargar los modelos y generar FRENTE + ESPALDA con una sola línea.

    from pokegan.demo import cargar_modelos, generar, variar_tipo
    cargar_modelos("pokegan_colab/modelos")
    generar("fuego", "volador", semilla=0, cantidad=5, modelo="vision")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import types as T
from .models.generator import Generator
from .postprocess import clean_pair
from .refine import truncate
from .sample import make_y, make_z, render

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELOS: dict[str, tuple] = {}


def cargar_modelos(carpeta) -> dict:
    """Carga cada modelos/<nombre>.pt (G_ema exportado por scripts/export_colab.py)."""
    for f in sorted(Path(carpeta).glob("*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        G = Generator(**ck["cfg"]["G"])
        G.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["G_ema"].items()})
        MODELOS[f.stem] = (G.to(DEVICE).eval(), ck)
        print(f"modelo '{f.stem}': {ck['label']} · paso {ck['step']}")
    print("dispositivo:", DEVICE)
    return MODELOS


def _tipos(tipo1, tipo2):
    tipos = [tipo1]
    if tipo2 and str(tipo2).lower() not in ("ninguno", "none", "") and T.normalize_type(tipo2) != T.normalize_type(tipo1):
        tipos.append(tipo2)
    return tipos


def mostrar(pares: np.ndarray, titulo: str = "", nombres=None) -> None:
    """Cada columna = una criatura: arriba FRENTE, abajo ESPALDA."""
    n = len(pares)
    fig, ax = plt.subplots(2, n, figsize=(2.1 * n, 4.6), squeeze=False)
    for k in range(n):
        for v, vista in enumerate(["FRENTE", "ESPALDA"]):
            ax[v, k].imshow(pares[k, v], interpolation="nearest")
            ax[v, k].set_xticks([])
            ax[v, k].set_yticks([])
            if k == 0:
                ax[v, k].set_ylabel(vista, fontsize=11)
        ax[0, k].set_title(nombres[k] if nombres else f"criatura {k + 1}", fontsize=10)
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()


def generar(tipo1, tipo2=None, semilla=0, cantidad=5, modelo="vision", fidelidad=False, pixel_art=False):
    """tipo1 / tipo2 en español o inglés ("fuego", "Volador", "water"...); tipo2=None o "ninguno" → un solo tipo.

    semilla: cambia el número → criaturas distintas (misma semilla → mismas criaturas).
    fidelidad=True: truncamiento de z0–z2 (siluetas más limpias, algo menos variedad).
    pixel_art=True: post-proceso declarado (paleta conjunta de 10 colores, fondo limpio, sin fragmentos).
    """
    G, ck = MODELOS[modelo]
    tipos = _tipos(tipo1, tipo2)
    z = make_z(cantidad, G.z_dim, semilla)
    if fidelidad:
        z = truncate(z, 1.0, chunks=(0, 1, 2), generator=torch.Generator().manual_seed(semilla))
    y = make_y(tipos, cantidad, semilla)
    pares = render(G, z, y, lam=ck["lam"], device=DEVICE)  # uint8 [cantidad, 2, 96, 96, 3]
    if pixel_art:
        pares = np.stack([np.stack(clean_pair(p[0], p[1], colors=10, keep_frac=0.15)) for p in pares])
    mostrar(pares, f"{T.label(T.encode(tipos))} · modelo {modelo} · semilla {semilla}"
            + (" · fidelidad" if fidelidad else "") + (" · pixel art" if pixel_art else ""))
    return pares


def variar_tipo(semilla=7, tipos=("fuego", "agua", "planta", "electrico", "hielo", "fantasma"), modelo="vision"):
    """Un MISMO z (la misma 'criatura') con distintos tipos: z fija el cuerpo, el tipo fija la paleta."""
    G, ck = MODELOS[modelo]
    z = make_z(1, G.z_dim, semilla).repeat(len(tipos), 1)
    y = torch.from_numpy(np.stack([T.encode([t]) for t in tipos]))
    pares = render(G, z, y, lam=ck["lam"], device=DEVICE)
    mostrar(pares, f"Mismo z (semilla {semilla}) con distintos tipos · modelo {modelo}",
            nombres=[T.label(T.encode([t])) for t in tipos])
    return pares
