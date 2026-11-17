"""Construye la caché de datos (data/cache/pairs96.npz) a partir de pairs.csv y las imágenes.

Uso:
    python scripts/prepare_data.py                      # rutas por defecto
    python scripts/prepare_data.py --root data/Pokemon  # carpeta con pairs.csv e images96/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan.data import build_cache, load_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT / "data" / "Pokemon"))
    ap.add_argument("--out", default=str(ROOT / "data" / "cache" / "pairs96.npz"))
    a = ap.parse_args()
    info = build_cache(Path(a.root) / "pairs.csv", a.root, a.out)
    full = load_cache(a.out, drop_gray=False)
    kept = load_cache(a.out, drop_gray=True)
    print(f"OK: {info['n']} pares -> {info['path']}")
    print(f"Pares en gris (descartados por defecto): {int(full.is_gray.sum())}; quedan {len(kept)}")


if __name__ == "__main__":
    main()
