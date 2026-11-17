"""Entrena PokeGAN.

Ejemplos:
    python scripts/train.py --config configs/main.yaml
    python scripts/train.py --config configs/main.yaml --benchmark 200      # mide it/s y memoria
    python scripts/train.py --config configs/main.yaml --iters 30000 --max-hours 4
    python scripts/train.py --config configs/main.yaml --resume auto         # reanuda runs/main/last.pt
    python scripts/train.py --config configs/smoke.yaml                      # prueba de humo en CPU
    python scripts/train.py --resume auto --iters 28000 --lr-g 1e-5 --lr-d 4e-5   # recocido final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "main.yaml"))
    ap.add_argument("--name", help="nombre del run (carpeta en runs/)")
    ap.add_argument("--resume", help="ruta a un checkpoint o 'auto' (= runs/<name>/last.pt)")
    ap.add_argument("--iters", type=int, help="número total de pasos")
    ap.add_argument("--max-hours", type=float, help="detener tras estas horas (guarda checkpoint)")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--benchmark", type=int, help="ejecutar N pasos sin guardar y reportar it/s")
    ap.add_argument("--memory-fraction", type=float, help="tope de VRAM (0-1) si se comparte la GPU")
    ap.add_argument("--lr-g", type=float, help="learning rate de G (p. ej. recocido final: 1e-5)")
    ap.add_argument("--lr-d", type=float, help="learning rate de D (p. ej. recocido final: 4e-5)")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.name:
        cfg["name"] = a.name
    if a.iters:
        cfg["train"]["iters"] = a.iters
    if a.max_hours:
        cfg["train"]["max_hours"] = a.max_hours
    if a.batch:
        cfg["train"]["batch"] = a.batch
    if a.device:
        cfg["device"] = a.device
    if a.memory_fraction:
        cfg["memory_fraction"] = a.memory_fraction
    if a.lr_g:
        cfg["train"]["lr_g"] = a.lr_g
    if a.lr_d:
        cfg["train"]["lr_d"] = a.lr_d
    if a.benchmark:
        cfg["train"]["benchmark"] = a.benchmark
        cfg["name"] = cfg["name"] + "_benchmark"
    out = Path(cfg.get("out_dir", "runs"))
    if not out.is_absolute():
        cfg["out_dir"] = str(ROOT / out)
    cache = Path(cfg["data"]["cache"])
    if not cache.is_absolute():
        cfg["data"]["cache"] = str(ROOT / cache)
    if a.resume:  # después de volver absolutas las rutas: funciona desde cualquier carpeta
        cfg["resume"] = str(Path(cfg["out_dir"]) / cfg["name"] / "last.pt") if a.resume == "auto" else a.resume
    train(cfg)


if __name__ == "__main__":
    main()
