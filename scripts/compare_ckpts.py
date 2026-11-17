"""Compara varios checkpoints con las MISMAS condiciones y semillas (selección del modelo final).

Métricas por checkpoint: KID×10³ frente/espalda (Inception, solo evaluación), mAP y acierto del conjunto de tipo
(evaluador DINOv3 linear probe), coherencia de paleta, diversidad L1 (dos z por condición) y fragmentos por vista.

Uso:
    python scripts/compare_ckpts.py runs/main/snap_015000.pt runs/main/snap_020000.pt runs/main/ckpt_024000.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import evaluate as E  # noqa: E402
from pokegan import refine as R  # noqa: E402
from pokegan.sample import load_generator  # noqa: E402
from pokegan.type_classifier import load_classifier, predict_logits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--clf", default=str(ROOT / "runs" / "type_probe" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "seleccion_checkpoint.md"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    clf, _ = load_classifier(a.clf, dev)
    real = E.real_eval_features(device=dev)
    rows, res = [], {}
    for path in a.ckpts:
        G, info = load_generator(path, ema=True, device=dev)
        g = torch.Generator().manual_seed(0)
        bank = info["label_bank"]
        y = bank[torch.randint(len(bank), (a.n,), generator=g)].float()
        half = a.n // 2
        y[half:] = y[:half]
        z = torch.randn(a.n, G.z_dim, generator=g)
        with torch.no_grad():
            x = torch.cat([((G(z[i:i + 128].to(dev), y[i:i + 128].to(dev), info["lam"]).clamp(-1, 1) + 1) * 127.5)
                           .round().to(torch.uint8).cpu() for i in range(0, a.n, 128)])
        r = {"step": info["step"]}
        for view, sl in (("front", slice(0, 3)), ("back", slice(3, 6))):
            r[f"kid_{view}"] = 1e3 * E.kid_from_feats(real[view], E.inception_features(x[:, sl], dev), 1000, 50)[0]
        tm = E.type_metrics(predict_logits(clf, torch.cat([x[:, :3], x[:, 3:]])), torch.cat([y, y]))
        r.update(type_map=tm["mAP"], set_acc=tm["set_acc"])
        xn = x.permute(0, 2, 3, 1).numpy()
        r["coherence"] = float(E.palette_hi_batch(xn[..., :3], xn[..., 3:]).mean())
        r["div_l1"] = float((x[:half].float() - x[half:].float()).abs().mean() / 255)
        frag = torch.cat([R.count_fragments(x[:, :3]), R.count_fragments(x[:, 3:])]).float()
        r["fragments"], r["single_body"] = float(frag.mean()), float((frag == 1).float().mean())
        fg = (x.view(len(x), 2, 3, 96, 96).min(2).values < 235).sum((-1, -2)).min(1).values
        r["empty_240"] = float((fg < 240).float().mean())  # fuera del soporte real (ningún sprite real < 240 px)
        res[path] = r
        rows.append(f"| `{Path(path).name}` | {r['step']} | {r['kid_front']:.1f} | {r['kid_back']:.1f} | "
                    f"{r['type_map']:.3f} | {r['set_acc']:.3f} | {r['coherence']:.3f} | {r['div_l1']:.4f} | "
                    f"{r['fragments']:.2f} | {100 * r['single_body']:.0f} % | {100 * r['empty_240']:.1f} % |")
        print(rows[-1], flush=True)
    md = ["| Checkpoint | Paso | KID×10³ frente ↓ | KID×10³ espalda ↓ | mAP tipo ↑ | Acierto conj. ↑ | Coherencia ↑ | "
          "Diversidad L1 ↑ | Fragmentos/vista ↓ | % cuerpo único ↑ | % vacíos (<240 px) ↓ |", "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(f"Comparación de checkpoints ({a.n} pares, mismas condiciones y semillas)\n\n" + "\n".join(md) + "\n",
                           encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
