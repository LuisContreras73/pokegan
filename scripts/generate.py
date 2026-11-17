"""Genera Pokémon nuevos (frente y espalda) con el generador entrenado.

Ejemplos:
    python scripts/generate.py --types fuego volador
    python scripts/generate.py --types hada fuego --n 16 --seed 3      # combinación no vista
    python scripts/generate.py --n 12                                  # tipos aleatorios
    python scripts/generate.py --types agua --truncation 0.7 --post
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import types as T  # noqa: E402
from pokegan.figures import POST_CAPTION, save_pairs_grid  # noqa: E402
from pokegan.postprocess import clean_pairs  # noqa: E402
from pokegan.sample import load_generator, make_y, make_z, pair_canvas, render, upscale  # noqa: E402


def _slug(text: str) -> str:
    """'Volador/Fuego' -> 'volador-fuego' (sin tildes, apto para nombres de archivo)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return ascii_text.lower().replace("/", "-")


def _refine_z(a, G, info, z, y):
    """Aplica el muestreador guiado por D elegido y devuelve los z refinados (en CPU)."""
    import json

    import torch

    from pokegan import refine as R
    from pokegan.models import Discriminator

    dev = next(G.parameters()).device
    torch.manual_seed(a.seed)
    if a.refinar == "trunc_grueso":
        return R.truncate(z, a.truncation or 0.7, chunks=(0, 1, 2), generator=torch.Generator().manual_seed(a.seed))
    beta = a.beta
    cal = ROOT / "results" / "refinamiento" / "metrics.json"
    if beta is None:
        beta = json.loads(cal.read_text())["_calibration"]["beta"] if cal.exists() else 1.0
    ck = torch.load(a.d_ckpt, map_location="cpu", weights_only=False)
    D = Discriminator(**ck["cfg"]["D"]).to(dev)
    D.load_state_dict(ck["D"])
    zd, yd = z.to(dev), torch.as_tensor(y).float().to(dev)
    if a.refinar == "sir":
        out = R.sir(G, D, yd, 8, info["lam"], beta, 4, torch.Generator().manual_seed(a.seed))
    else:
        out = R.dgflow(G, D, zd, yd, info["lam"], steps=25, beta=beta,
                       chunks=(0, 1, 2) if a.refinar == "dgflow_grueso" else None)
    print(f"Refinamiento '{a.refinar}' con D de {a.d_ckpt} (paso {ck['step']}), β = {beta:.3f}")
    return out.cpu()


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "main" / "last.pt"),
                    help="checkpoint .pt (por defecto runs/main/last.pt)")
    ap.add_argument("--types", nargs="+", metavar="TIPO",
                    help="1 o 2 tipos en español o inglés (p. ej. fuego volador). "
                         "Si se omite, cada Pokémon recibe una condición aleatoria")
    ap.add_argument("--n", type=int, default=8, help="cantidad de Pokémon (por defecto 8)")
    ap.add_argument("--seed", type=int, default=0,
                    help="semilla: la misma semilla da los mismos Pokémon (por defecto 0)")
    ap.add_argument("--truncation", type=float, default=None,
                    help="truco de truncamiento: re-muestrea |z| > valor (p. ej. 0.7). "
                         "Menor = más típico y nítido, pero menos diverso")
    ap.add_argument("--post", action="store_true",
                    help="guarda además la versión post-procesada (fondo blanco + paleta conjunta)")
    ap.add_argument("--colors", type=int, default=15, help="colores de la paleta con --post")
    ap.add_argument("--out", default=str(ROOT / "results" / "generados"),
                    help="carpeta de salida (por defecto results/generados)")
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                    help="usar pesos EMA (por defecto); --no-ema usa el G crudo")
    ap.add_argument("--device", default=None, help="cuda o cpu (por defecto cuda si existe)")
    ap.add_argument("--cols", type=int, default=4, help="Pokémon por fila en la grilla")
    ap.add_argument("--refinar", choices=["no", "trunc_grueso", "sir", "dgflow_grueso", "dgflow"], default="no",
                    help="muestreo guiado por NUESTRO D (docs/matematica.md §10b): truncamiento solo en z0–z2, "
                         "remuestreo por importancia, o flujo DGflow-KL en z (todo o solo z0–z2)")
    ap.add_argument("--d-ckpt", default=str(ROOT / "runs" / "main" / "last.pt"),
                    help="checkpoint COMPLETO con D (para --refinar sir/dgflow)")
    ap.add_argument("--beta", type=float, default=None,
                    help="temperatura β del logit de D (por defecto la calibrada en results/refinamiento/metrics.json, o 1)")
    a = ap.parse_args(argv)

    types = None
    if a.types:
        types = " ".join(a.types).replace("/", " ").replace(",", " ").split()
        try:
            T.encode(types)
        except ValueError as e:
            ap.error(str(e))
    if not Path(a.ckpt).exists():
        ap.error(f"no existe el checkpoint {a.ckpt}")

    G, info = load_generator(a.ckpt, ema=a.ema, device=a.device)
    z = make_z(a.n, G.z_dim, a.seed, None if a.refinar == "trunc_grueso" else a.truncation)
    y = make_y(types, a.n, a.seed, info["label_bank"])
    if a.refinar != "no":
        z = _refine_z(a, G, info, z, y)
    pairs = render(G, z, y, lam=info["lam"])
    labels = [T.label(v.numpy()) for v in y]

    name = _slug(labels[0]) if types else "aleatorio"
    run = Path(a.out) / f"{name}_semilla{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    title = (f"PokeGAN · {labels[0] if types else 'tipos aleatorios'} · semilla {a.seed} · "
             f"paso {info['step']}" + ("" if a.ema else " · sin EMA")
             + ("" if a.refinar == "no" else f" · muestreo: {a.refinar}"
                + (f" {a.truncation or 0.7}" if a.refinar == "trunc_grueso" else "")))
    cell_labels = [f"#{i} {lab}" for i, lab in enumerate(labels)]
    save_pairs_grid(pairs, run / "grilla.png", ncols=a.cols, labels=cell_labels, title=title)

    versions = [("", pairs)]
    if a.post:
        post = clean_pairs(pairs, colors=a.colors)
        versions.append(("_post", post))
        save_pairs_grid(post, run / "grilla_post.png", ncols=a.cols, labels=cell_labels,
                        title=title + " · post-procesado", caption=POST_CAPTION.format(c=a.colors))
    for suffix, batch in versions:
        for i, p in enumerate(batch):
            Image.fromarray(p[0]).save(run / f"{i:02d}_frente{suffix}.png")
            Image.fromarray(p[1]).save(run / f"{i:02d}_espalda{suffix}.png")
            Image.fromarray(upscale(pair_canvas(p), 4)).save(run / f"{i:02d}_par{suffix}_x4.png")

    print(f"Checkpoint: {a.ckpt} (paso {info['step']}, lam={info['lam']:.2f}, "
          f"{'EMA' if a.ema else 'G crudo'})")
    print(f"{a.n} Pokemon generados en: {run}")
    print(f"  grilla:      {run / 'grilla.png'}" + (f"  y  {run / 'grilla_post.png'}" if a.post else ""))
    print("  individuales: NN_frente.png / NN_espalda.png (96x96) y NN_par_x4.png (vista ampliada)")
    return run


if __name__ == "__main__":
    main()
