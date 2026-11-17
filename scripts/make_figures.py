"""Genera las figuras de resultados a partir de un checkpoint.

Ejemplo:
    python scripts/make_figures.py --ckpt runs/main/best.pt --out results/figuras

Figuras (salida cruda de G salvo la 07, que la compara con la post-procesada):
    01 monotipos (18 tipos x n_z)         05 interpolación en z (PNG + GIF)
    02 12 combinaciones dobles frecuentes  06 interpolación de tipo Fuego -> Agua (PNG + GIF)
    03 las 11 combinaciones NO vistas      07 crudo vs post-procesado
    04 mismo z, los 18 tipos               08 curvas de entrenamiento (log.csv del run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import figures as F  # noqa: E402
from pokegan import types as T  # noqa: E402
from pokegan.sample import load_generator  # noqa: E402


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "main" / "last.pt"),
                    help="checkpoint .pt (por defecto runs/main/last.pt)")
    ap.add_argument("--out", default=str(ROOT / "results" / "figuras"), help="carpeta de salida")
    ap.add_argument("--seed", type=int, default=0, help="semilla de los z (por defecto 0)")
    ap.add_argument("--n-z", type=int, default=6, help="z distintos por condición (por defecto 6)")
    ap.add_argument("--truncation", type=float, default=None, help="truco de truncamiento (opcional)")
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                    help="usar pesos EMA (por defecto); --no-ema usa el G crudo")
    ap.add_argument("--log", default=None, help="log.csv (por defecto, el de la carpeta del ckpt)")
    ap.add_argument("--device", default=None, help="cuda o cpu (por defecto cuda si existe)")
    ap.add_argument("--scale", type=int, default=2, help="ampliación de los sprites (por defecto 2)")
    a = ap.parse_args(argv)
    if not Path(a.ckpt).exists():
        ap.error(f"no existe el checkpoint {a.ckpt}")

    G, info = load_generator(a.ckpt, ema=a.ema, device=a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(lam=info["lam"], truncation=a.truncation)
    s, nz, sc = a.seed, a.n_z, a.scale
    step = f"paso {info['step']}"
    paths = [
        F.grid_by_type(G, [(t,) for t in T.TYPES], nz, s, out / "01_monotipos.png", scale=sc,
                       title=f"Los 18 monotipos · columnas = z distintos · frente | espalda · {step}",
                       **kw),
    ]
    if info["label_bank"] is not None:
        paths.append(F.grid_by_type(
            G, F.frequent_duals(info["label_bank"], 12), nz, s, out / "02_duales_frecuentes.png",
            scale=sc, title=f"12 combinaciones dobles frecuentes · {step}", **kw))
    paths += [
        F.grid_by_type(G, F.UNSEEN, nz, s, out / "03_combos_no_vistos.png", scale=sc,
                       title="Combinaciones NO vistas en entrenamiento (generalización "
                             f"composicional: e = suma de embeddings) · {step}", **kw),
        F.same_z_all_types(G, min(nz, 6), s, out / "04_mismo_z_todos_los_tipos.png", scale=sc,
                           **kw),
        F.interpolation_strip(G, ["fire"], ["fire"], s, s + 1, out / "05_interp_z.png",
                              scale=sc, **kw),
        F.interpolation_gif(G, ["fire"], ["fire"], s, s + 1, out / "05_interp_z.gif", **kw),
        F.interpolation_strip(G, ["fire"], ["water"], s, s, out / "06_interp_tipo_fuego_agua.png",
                              scale=sc, **kw),
        F.interpolation_gif(G, ["fire"], ["water"], s, s, out / "06_interp_tipo_fuego_agua.gif",
                            **kw),
        F.raw_vs_post(G, [("fire",), ("water",), ("grass", "poison"), ("dragon", "fairy")], s,
                      out / "07_crudo_vs_post.png", n_z=2, scale=sc, **kw),
    ]
    log = Path(a.log) if a.log else Path(a.ckpt).parent / "log.csv"
    if log.exists():
        paths.append(F.training_curves(log, out / "08_curvas_entrenamiento.png"))
    else:
        print(f"(no se encontró {log}: se omiten las curvas de entrenamiento)")

    print(f"Figuras generadas con {a.ckpt} ({step}, {'EMA' if a.ema else 'G crudo'}):")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    main()
