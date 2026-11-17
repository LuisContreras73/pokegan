"""Entrena el clasificador de tipos DESDE CERO (solo para evaluar el GAN) y lo guarda.

Frentes y espaldas reales de las especies de entrenamiento como muestras separadas; parada
temprana y umbrales por clase en las especies retenidas (validación).

Uso:
    python scripts/train_type_classifier.py                       # runs/type_classifier/best.pt
    python scripts/train_type_classifier.py --epochs 30 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan.data import load_curated  # noqa: E402
from pokegan.type_classifier import train_type_classifier  # noqa: E402
from pokegan.types import TYPES_ES  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "cache" / "pairs96.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "type_classifier"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--device", default=None, help="cuda / cpu (por defecto cuda si está disponible)")
    a = ap.parse_args(argv)

    train, val = load_curated(a.data)
    print(f"Entrenamiento: {len(train)} pares ({2 * len(train)} imágenes); "
          f"validación: {len(val)} pares de especies retenidas")
    m = train_type_classifier(train, val, a.out, epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd,
                              device=a.device)

    print("\nMétricas en validación (especies retenidas):")
    print(f"  mAP macro      {m['mAP']:.3f}")
    print(f"  F1 macro       {m['f1_macro']:.3f}")
    print(f"  set-accuracy   {m['set_acc']:.3f}")
    print("  recall por tipo:")
    for t, r in sorted(m["per_type_recall"].items(), key=lambda kv: -kv[1]):
        print(f"    {TYPES_ES[t]:<10s} {r:.3f}")
    (Path(a.out) / "val_metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {Path(a.out) / 'best.pt'}")


if __name__ == "__main__":
    main()
