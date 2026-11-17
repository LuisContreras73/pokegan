"""Exporta los generadores publicados en weights/ (G_ema en fp16, ~17 MB cada uno).

Cada archivo trae lo necesario para muestrear y evaluar sin el checkpoint de entrenamiento:
  G_ema            pesos del generador promediado (EMA), con las estadísticas de BatchNorm ya recalculadas
  cfg              hiperparámetros de la arquitectura
  lam, step        peso del tipo al final de la transición y paso de entrenamiento
  label_bank       condiciones del conjunto de entrenamiento (para muestrear tipos con su frecuencia real)
  standing_applied True: load_generator no vuelve a recalcular BatchNorm

Uso:
    python scripts/export_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan.sample import load_generator  # noqa: E402

MODELS = {
    "pokegan_pura": ("Pura: generador y discriminador desde cero", ROOT / "runs" / "main" / "final.pt"),
    "pokegan_vision": ("Vision: discriminador auxiliar ConvNeXt-T durante el entrenamiento", ROOT / "runs" / "main" / "final_vision.pt"),
}


def main() -> None:
    torch.set_num_threads(4)
    out = ROOT / "weights"
    out.mkdir(exist_ok=True)
    for key, (label, path) in MODELS.items():
        G, info = load_generator(path, ema=True, device="cpu", standing=True, standing_seed=0)
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in G.state_dict().items()}
        bank = info["label_bank"]
        bank = None if bank is None else np.asarray(bank, dtype=np.uint8)
        torch.save({"G_ema": sd, "cfg": {"G": info["cfg"]["G"]}, "lam": info["lam"], "step": info["step"],
                    "label_bank": bank, "label": label, "standing_applied": True}, out / f"{key}.pt")
        print(f"  {key}.pt: paso {info['step']} · {(out / f'{key}.pt').stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
