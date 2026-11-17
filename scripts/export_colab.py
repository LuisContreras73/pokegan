"""Empaqueta PokeGAN para inferencia en Google Colab: código + generadores livianos + notebook.

Crea colab/pokegan_colab.zip con:
  pokegan/                    el paquete (src/pokegan)
  modelos/<clave>.pt          G_ema con standing stats YA aplicadas (eval), cfg, λ, step  (fp16, ~17 MB c/u)
Y el notebook notebooks/PokeGAN_Colab.ipynb (subirlo a Colab junto con el zip).

Uso:
    python scripts/export_colab.py
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan.sample import load_generator  # noqa: E402

MODELS = {
    "pura": ("Entrenado desde cero", ROOT / "weights" / "pokegan_pura.pt"),
    "vision": ("Desde cero + discriminador auxiliar ConvNeXt-T", ROOT / "weights" / "pokegan_vision.pt"),
}


def main() -> None:
    torch.set_num_threads(4)
    out_dir = ROOT / "colab"
    stage = out_dir / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "modelos").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "pokegan", stage / "pokegan", ignore=shutil.ignore_patterns("__pycache__"))
    for key, (label, path) in MODELS.items():
        if not path.exists():
            continue
        G, info = load_generator(path, ema=True, device="cpu", standing=True, standing_seed=0)
        cfg = {"G": info["cfg"]["G"], "D": info["cfg"]["D"]}
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in G.state_dict().items()}  # fp16: zip a la mitad
        torch.save({"G_ema": sd, "cfg": cfg, "lam": info["lam"], "step": info["step"], "label": label,
                    "standing_applied": True}, stage / "modelos" / f"{key}.pt")
        print(f"  {key}: paso {info['step']} ({label})")
    zpath = out_dir / "pokegan_colab.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in stage.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(stage))
    shutil.rmtree(stage)
    nb = ROOT / "notebooks" / "PokeGAN_Colab.ipynb"
    if nb.exists():
        shutil.copy2(nb, out_dir / nb.name)
    print(f"Listo: {zpath} ({zpath.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
