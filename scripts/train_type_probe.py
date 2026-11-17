"""Entrena el evaluador de tipos con *linear probe* (backbone preentrenado CONGELADO; solo evaluación).

Compara DINOv3 y ConvNeXt en las especies retenidas y guarda el mejor en runs/type_probe/best.pt.
Luego: python scripts/evaluate.py --ckpt runs/main/best.pt --clf runs/type_probe/best.pt

Uso:
    python scripts/train_type_probe.py
    python scripts/train_type_probe.py --backbones dinov3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan.data import load_curated  # noqa: E402
from pokegan.evaluate import type_metrics  # noqa: E402
from pokegan.type_classifier import tune_thresholds, views_as_samples  # noqa: E402
from pokegan.type_probe import Backbone, extract, fit_head, save_probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "cache" / "pairs96.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "type_probe"))
    ap.add_argument("--backbones", nargs="+", default=["dinov3", "convnext"])
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

    tr, va = load_curated(a.data)
    x_tr, y_tr = views_as_samples(tr)
    x_va, y_va = views_as_samples(va)
    y_tr, y_va = torch.as_tensor(y_tr), torch.as_tensor(y_va)
    print(f"train {len(x_tr)} imágenes | val {len(x_va)} (especies retenidas) | {device}", flush=True)

    results, best = {}, None
    for name in a.backbones:
        bb = Backbone(name).to(device)
        f_tr = torch.cat([extract(bb, x_tr, device), extract(bb, x_tr, device, flip=True)])
        f_va = extract(bb, x_va, device)
        head, info = fit_head(f_tr, torch.cat([y_tr, y_tr]), f_va, y_va, device)
        with torch.no_grad():
            probs = torch.sigmoid(head(f_va)).numpy()
        th = tune_thresholds(probs, y_va.numpy())
        m = type_metrics(torch.logit(torch.as_tensor(probs).clamp(1e-6, 1 - 1e-6)), y_va, th)
        results[name] = {k: m[k] for k in ("mAP", "f1_macro", "set_acc")}
        print(f"[{name}] val mAP {m['mAP']:.3f} | F1 {m['f1_macro']:.3f} | set_acc {m['set_acc']:.3f}", flush=True)
        if best is None or m["mAP"] > best[1]["mAP"]:
            best = (name, m, head, th)
        del bb
        torch.cuda.empty_cache()

    name, m, head, th = best
    out = Path(a.out)
    save_probe(out / "best.pt", name, head, th, {k: v for k, v in m.items() if k != "per_type_recall"})
    (out / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Mejor backbone: {name} (mAP {m['mAP']:.3f}) → {out / 'best.pt'}")
    print("Recall por tipo:", json.dumps({k: round(v, 2) for k, v in m["per_type_recall"].items()}))


if __name__ == "__main__":
    main()
