"""Figuras y animaciones del README, reproducibles a partir de los pesos publicados (weights/).

Genera en results/media/:
  paseo_latente.gif      cada columna recorre un círculo máximo en z (frente arriba, espalda abajo, mismo z)
  transicion_tipos.gif   z fijo; la condición se interpola entre tipos (el embedding es lineal en y)
  trozos_latente.png     qué controla cada trozo z0..z5: se re-muestrea un trozo y el resto queda fijo
  entrenamiento.gif      evolución de muestras fijas durante el entrenamiento (si existe runs/main/samples/)

Uso:
    python scripts/make_readme_media.py [--model vision|pura] [--device cpu]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import types as T  # noqa: E402
from pokegan.sample import load_generator, render  # noqa: E402

OUT = ROOT / "results" / "media"
INK, MUTED, PAPER = (29, 43, 36), (93, 107, 99), (255, 255, 255)
TYPE_COLORS = {
    "normal": "#A8A77A", "fighting": "#C22E28", "flying": "#A98FF3", "poison": "#A33EA1", "ground": "#E2BF65",
    "rock": "#B6A136", "bug": "#A6B91A", "ghost": "#735797", "steel": "#B7B7CE", "fire": "#EE8130", "water": "#6390F0",
    "grass": "#7AC74C", "electric": "#F7D02C", "psychic": "#F95587", "ice": "#96D9D6", "dragon": "#6F35FC",
    "dark": "#705746", "fairy": "#D685AD",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def up(img: np.ndarray, k: int) -> Image.Image:
    return Image.fromarray(img).resize((img.shape[1] * k, img.shape[0] * k), Image.NEAREST)


def onehot(types: list[str]) -> torch.Tensor:
    return torch.from_numpy(T.encode(types)).float()


def save_gif(frames: list[Image.Image], path: Path, ms: int) -> None:
    # paleta común construida con varios cuadros repartidos (con uno solo faltan colores de otros tipos)
    sample = frames[:: max(1, len(frames) // 16)]
    mosaic = Image.new("RGB", (frames[0].width, frames[0].height * len(sample)))
    for i, f in enumerate(sample):
        mosaic.paste(f.convert("RGB"), (0, i * frames[0].height))
    pal = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, optimize=True, disposal=1)
    print(f"  {path.relative_to(ROOT)}  ({len(frames)} cuadros, {path.stat().st_size / 2**20:.1f} MB)")


def chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str, f) -> None:
    x, y = xy
    w = draw.textlength(text, font=f)
    draw.rounded_rectangle((x - w / 2 - 10, y - 13, x + w / 2 + 10, y + 13), radius=13, fill=color)
    draw.text((x, y), text, font=f, fill=(255, 255, 255), anchor="mm")


def pair_frame(pairs: np.ndarray, labels: list[tuple[str, str]], k: int, title: str, sub: str) -> Image.Image:
    """Lámina: fila de etiquetas, fila FRENTE, fila ESPALDA (cada columna = una criatura)."""
    n, s, pad, left, top = len(pairs), 96 * k, 14, 92, 92
    W, H = left + n * (s + pad) + pad, top + 2 * (s + pad) + 26
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    d.text((pad, 14), title, font=font(22, True), fill=INK)
    d.text((pad, 44), sub, font=font(14), fill=MUTED)
    for row, name in enumerate(["FRENTE", "ESPALDA"]):
        d.text((left - 12, top + row * (s + pad) + s / 2), name, font=font(13, True), fill=MUTED, anchor="rm")
    for c in range(n):
        x = left + c * (s + pad)
        chip(d, (x + s // 2, top - 16), labels[c][0], labels[c][1], font(14, True))
        for row in range(2):
            im.paste(up(pairs[c, row], k), (x, top + row * (s + pad)))
    return im


def paseo_latente(G, lam, device, n_frames=60) -> None:
    cols = [["fire"], ["water"], ["grass"], ["electric"], ["ghost"], ["dragon", "flying"]]
    g = torch.Generator().manual_seed(11)
    a, b = torch.randn(len(cols), 120, generator=g), torch.randn(len(cols), 120, generator=g)
    y = torch.stack([onehot(c) for c in cols])
    labels = [("/".join(T.TYPES_ES[t] for t in c), TYPE_COLORS[c[0]]) for c in cols]
    frames = []
    for i in range(n_frames):
        t = 2 * math.pi * i / n_frames
        z = math.cos(t) * a + math.sin(t) * b          # círculo máximo: cada punto sigue siendo N(0, I)
        pairs = render(G, z, y, lam=lam, device=device)
        frames.append(pair_frame(pairs, labels, 2, "Paseo por el espacio latente",
                                 "z recorre un círculo y el tipo queda fijo: frente y espalda cambian juntos porque salen del mismo z"))
    save_gif(frames, OUT / "paseo_latente.gif", 70)


def transicion_tipos(G, lam, device, steps=14, hold=6) -> None:
    ciclo = ["fire", "water", "grass", "electric", "ice", "psychic", "ghost", "dragon"]
    g = torch.Generator().manual_seed(23)
    z = torch.randn(4, 120, generator=g)
    frames = []
    for i, ta in enumerate(ciclo):
        tb = ciclo[(i + 1) % len(ciclo)]
        for s in [0.0] * hold + list(np.linspace(0, 1, steps, endpoint=False)):
            y = ((1 - s) * onehot([ta]) + s * onehot([tb])).repeat(4, 1)
            pairs = render(G, z, y, lam=lam, device=device)
            lab = T.TYPES_ES[ta] if s == 0 else f"{T.TYPES_ES[ta]} → {T.TYPES_ES[tb]}"
            col = TYPE_COLORS[ta] if s < 0.5 else TYPE_COLORS[tb]
            frames.append(pair_frame(pairs, [(lab, col)] * 4, 2, "Mismo z, distinto tipo",
                                     "cuatro criaturas con z fijo; solo se interpola el tipo: z decide sobre todo la forma; el tipo, sobre todo la paleta"))
    save_gif(frames, OUT / "transicion_tipos.gif", 80)


def trozos_latente(G, lam, device, model: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = torch.Generator().manual_seed(4)
    base = torch.randn(1, 120, generator=g)
    y1 = onehot(["fire"]).unsqueeze(0)
    ncol = 6
    rows, deltas = [], []
    for k in range(6):
        z = base.repeat(ncol, 1)
        z[1:, 20 * k:20 * (k + 1)] = torch.randn(ncol - 1, 20, generator=g)
        p = render(G, z, y1.repeat(ncol, 1), lam=lam, device=device)
        rows.append(p)
        deltas.append(float(np.abs(p[1:].astype(float) - p[:1].astype(float)).mean() / 255))
    fig, ax = plt.subplots(6, ncol, figsize=(ncol * 1.9 + 1.4, 6 * 1.05 + 0.9))
    for k in range(6):
        for c in range(ncol):
            ax[k, c].imshow(np.concatenate([rows[k][c, 0], rows[k][c, 1]], axis=1), interpolation="nearest")
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
            for sp in ax[k, c].spines.values():
                sp.set_edgecolor("#2b7bb9" if c == 0 else "#d0d0d0"); sp.set_linewidth(1.8 if c == 0 else 0.8)
        ax[k, 0].set_ylabel(f"$z_{k}$\nΔ = {deltas[k]:.3f}", rotation=0, ha="right", va="center", fontsize=11)
    ax[0, 0].set_title("referencia", fontsize=10, color="#2b7bb9")
    for c in range(1, ncol):
        ax[0, c].set_title(f"re-muestreo {c}", fontsize=10)
    fig.suptitle(f"¿Qué controla cada trozo del latente? (modelo {model}, tipo Fuego; cada celda: frente | espalda)\n"
                 "Fila k: se re-muestrea solo $z_k$ y el resto queda fijo. Δ = cambio medio de píxeles respecto a la referencia",
                 fontsize=11.5, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "trozos_latente.png", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"  results/media/trozos_latente.png  (Δ por trozo: {', '.join(f'{d:.3f}' for d in deltas)})")


def entrenamiento() -> None:
    src = ROOT / "runs" / "main" / "samples"
    steps = sorted(src.glob("step_*.png"))
    if not steps:
        print("  (sin runs/main/samples: se omite entrenamiento.gif)")
        return
    conds = (src / "conditions.txt").read_text(encoding="utf-8").split("\n")
    pick = [conds.index(c) for c in ["Fuego", "Agua", "Planta", "Eléctrico", "Fantasma"]]
    frames = []
    for p in steps:
        step = int(p.stem.split("_")[1])
        if step > 24000:
            continue
        im = Image.open(p).convert("RGB")
        cell_w, cell_h = im.width // 4, im.height // len([c for c in conds if c])
        strip = [im.crop((0, r * cell_h, 3 * cell_w, (r + 1) * cell_h)) for r in pick]
        W, H = 3 * cell_w + 130, len(pick) * cell_h + 74
        fr = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(fr)
        d.text((14, 12), f"Entrenamiento · paso {step:,}".replace(",", " "), font=font(20, True), fill=INK)
        d.text((14, 40), "mismas condiciones y semillas en cada paso (frente | espalda)", font=font(13), fill=MUTED)
        for i, (r, s) in enumerate(zip(pick, strip)):
            fr.paste(s, (124, 66 + i * cell_h))
            d.text((112, 66 + i * cell_h + cell_h / 2), conds[r], font=font(14, True), fill=INK, anchor="rm")
        frames.append(fr.resize((W * 3 // 4, H * 3 // 4), Image.LANCZOS))
    frames += [frames[-1]] * 8
    save_gif(frames, OUT / "entrenamiento.gif", 160)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vision", choices=["vision", "pura"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    G, info = load_generator(ROOT / "weights" / f"pokegan_{a.model}.pt", device=a.device)
    dev = next(G.parameters()).device
    torch.manual_seed(0)
    with torch.no_grad():
        paseo_latente(G, info["lam"], dev)
        transicion_tipos(G, info["lam"], dev)
        Gp, ip = load_generator(ROOT / "weights" / "pokegan_pura.pt", device=a.device)
        trozos_latente(Gp, ip["lam"], dev, "pura")
    entrenamiento()


if __name__ == "__main__":
    main()
