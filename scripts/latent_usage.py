"""¿Cuánto usa G cada trozo del latente? (subutilización del latente, no colapso de modos)

Para cada trozo k de z (z0..z5, 20 dims c/u): re-muestrear SOLO z_k (resto de z y tipo fijos) y medir el cambio en
la imagen relativo a re-muestrear z completo:
  cambio_rel(k) = E|G(z con z_k nuevo) − G(z)| / E|G(z nuevo) − G(z)|           (píxeles, ambas vistas)
  silueta(k)    = 1 − IoU de las máscaras de primer plano (min RGB < 235)
Si el modelo usara todo z, varios trozos tendrían cambio_rel alto; si solo usa z0, cambio_rel(z0) ≈ 1 y el resto ≈ 0.

Uso:
    python scripts/latent_usage.py weights/pokegan_pura.pt weights/pokegan_vision.pt --out results/uso_latente.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan.sample import load_generator  # noqa: E402


@torch.no_grad()
def usage(G, lam, bank, n=512, seed=0):
    dev = next(G.parameters()).device
    g = torch.Generator().manual_seed(seed)
    y = bank[torch.randint(len(bank), (n,), generator=g)].float().to(dev)
    z = torch.randn(n, G.z_dim, generator=g).to(dev)
    z_full = torch.randn(n, G.z_dim, generator=g).to(dev)
    x0 = G(z, y, lam)
    fg0 = (x0.view(n, 2, 3, 96, 96).min(2).values < 0.843)
    base = (G(z_full, y, lam) - x0).abs().mean()
    out = []
    for k in range(6):
        zk = z.clone()
        zk[:, 20 * k:20 * (k + 1)] = torch.randn(n, 20, generator=g).to(dev)
        xk = G(zk, y, lam)
        fgk = (xk.view(n, 2, 3, 96, 96).min(2).values < 0.843)
        inter = (fg0 & fgk).sum((-1, -2)).float()
        union = (fg0 | fgk).sum((-1, -2)).float().clamp_min(1)
        out.append(((xk - x0).abs().mean() / base).item())
        out.append((1 - inter / union).mean().item())
    return np.array(out).reshape(6, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "results" / "uso_latente.png"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {}
    for c in a.ckpts:
        G, info = load_generator(c, ema=True)
        res[Path(c).stem] = usage(G, info["lam"], info["label_bank"])
    names = [f"z{k}\n({['tallo 3×3', '6 px', '12 px', '24 px', '48 px', 'cabezas 96 px'][k]})" for k in range(6)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.8 / len(res)
    for i, (name, r) in enumerate(res.items()):
        for j, ax in enumerate(axes):
            ax.bar(np.arange(6) + i * w, r[:, j], width=w, label=name)
    axes[0].set_title("Cambio en la imagen al re-muestrear SOLO ese trozo\n(relativo a re-muestrear z completo = 1)")
    axes[1].set_title("Cambio de silueta (1 − IoU)")
    for ax in axes:
        ax.set_xticks(np.arange(6) + 0.4 - w / 2)
        ax.set_xticklabels(names, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Uso del latente por trozo: el modelo usa casi solo z0 (subutilización, no colapso de modos)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, facecolor="white")
    print("trozo | " + " | ".join(f"{n}: cambio, silueta" for n in res))
    for k in range(6):
        print(f"z{k} | " + " | ".join(f"{r[k, 0]:.3f}, {r[k, 1]:.3f}" for r in res.values()))
    print(f"Guardado en {a.out}")


if __name__ == "__main__":
    main()
