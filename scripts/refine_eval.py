"""Compara muestreo crudo vs refinamiento con NUESTRO D (y truncamiento), con las mismas condiciones y semillas.

Métodos (`--methods`), con parámetros tras ':':
  raw                      z ~ N(0, I)
  trunc:0.8                truncamiento global (BigGAN)
  trunc_coarse:0.7         truncamiento solo en z0–z2 (silueta 3×3, 6×6, 12×12)
  sir:8                    remuestreo por importancia con β = 1 entre 8 candidatos
  sircal:8                 ídem con β calibrado por Platt (especies no vistas vs falsos)
  dgflow:25                DGflow-KL 25 pasos (η=0.1, γ=0.01), β calibrado, todos los trozos
  dgflow_coarse:25         ídem moviendo solo z0–z2
  ddls:200                 Langevin latente (ε=0.01) con β calibrado
  support:240              restricción de soporte: re-muestrear z si una vista tiene < 240 px de primer plano
  support_trunc:240        ídem + truncamiento grueso 1.0
  trunc_sir:8:2            candidatos con truncamiento grueso 0.7 + selección por D (β calibrado, guía de tipo w=2)
  Un tercer campo fija w (guía de tipo) en sir/sircal/dgflow/trunc_sir, p. ej. sircal:8:2
Métricas: KID (Inception, SOLO evaluación), mAP de tipo (evaluador runs/type_probe/best.pt), coherencia de
paleta, diversidad (L1 entre 2 z con la misma condición), fragmentos por vista (componentes conexas), novedad
(fracción de frentes más cerca del entrenamiento que el 5 % más cercano de especies no vistas) y logit de D.

Uso:
    python scripts/refine_eval.py --g-ckpt runs/main/last.pt --d-ckpt runs/main/last.pt --n 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import evaluate as E  # noqa: E402
from pokegan import refine as R  # noqa: E402
from pokegan import types as T  # noqa: E402
from pokegan.data import load_curated, to_tensor_pairs  # noqa: E402
from pokegan.models import Discriminator  # noqa: E402
from pokegan.sample import load_generator  # noqa: E402
from pokegan.type_classifier import load_classifier, predict_logits  # noqa: E402

SHOW = [("fire",), ("water",), ("grass",), ("electric",), ("ghost",), ("dragon",), ("fire", "flying"), ("fire", "fairy")]
COARSE = (0, 1, 2)


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def render(G, z, y, lam, bs=64):
    with torch.no_grad():
        return torch.cat([to_u8(G(z[i:i + bs], y[i:i + bs], lam)).cpu() for i in range(0, len(z), bs)])


def batched(fn, z, y, bs=64, **kw):
    return torch.cat([fn(z=z[i:i + bs], y=y[i:i + bs], **kw) for i in range(0, len(z), bs)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g-ckpt", default=str(ROOT / "runs" / "main" / "last.pt"))
    ap.add_argument("--d-ckpt", default=str(ROOT / "runs" / "main" / "last.pt"))
    ap.add_argument("--clf", default=str(ROOT / "runs" / "type_probe" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "refinamiento"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--methods", nargs="+",
                    default=["raw", "trunc_coarse:0.7", "trunc:0.8", "sircal:8", "dgflow_coarse:25", "dgflow:25"])
    ap.add_argument("--n-aug", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    G, info = load_generator(a.g_ckpt, ema=True, device=dev)
    lam = info["lam"]
    ckd = torch.load(a.d_ckpt, map_location="cpu", weights_only=False)
    D = Discriminator(**ckd["cfg"]["D"]).to(dev)
    D.load_state_dict(ckd["D"])
    D.eval().requires_grad_(False)
    clf, _ = load_classifier(a.clf, dev)
    real = E.real_eval_features(device=dev)
    bank = info["label_bank"]
    print(f"G: {a.g_ckpt} (paso {info['step']}) | D: {a.d_ckpt} (paso {ckd['step']}) | λ={lam}", flush=True)

    # --- referencia de novedad: distancia al vecino más cercano de especies NO vistas
    tr, va = load_curated(ROOT / "data" / "cache" / "pairs96.npz")
    train_f = tr.front
    _, d_val = E.nearest_neighbors(va.front, train_f, k=1, device=dev)
    q05 = float(np.quantile(np.asarray(d_val)[:, 0], 0.05))

    # --- calibración de Platt del logit de D: reales de validación vs falsos con las mismas etiquetas
    xv = to_tensor_pairs(va, "cpu")
    yv = torch.from_numpy(va.y).float().to(dev)
    zf = torch.randn(len(yv), G.z_dim, generator=torch.Generator().manual_seed(99)).to(dev)
    xf = render(G, zf, yv, lam)
    beta_cal, b_cal = R.platt_beta(D, xv.to(dev).float() / 127.5 - 1, yv, xf.to(dev).float() / 127.5 - 1, yv)
    print(f"Platt: P(real|s) = σ({beta_cal:.3f}·s + {b_cal:.3f})  ⇒ β calibrado = {beta_cal:.3f}", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    y = bank[torch.randint(len(bank), (a.n,), generator=g)].float()
    half = a.n // 2
    y[half:2 * half] = y[:half]  # dos z por condición ⇒ diversidad intra-condición
    z0 = torch.randn(a.n, G.z_dim, generator=g)
    yd, zd = y.to(dev), z0.to(dev)
    ys = torch.from_numpy(np.stack([T.encode(list(c)) for c in SHOW])).repeat_interleave(4, 0).to(dev)
    zs = torch.randn(len(ys), G.z_dim, generator=torch.Generator().manual_seed(7)).to(dev)

    results, panels = {"_calibration": {"beta": beta_cal, "bias": b_cal, "novelty_q05_val": q05}}, {}
    for spec in a.methods:
        name, _, rest = spec.partition(":")
        arg, _, rest2 = rest.partition(":")
        wstr, _, etastr = rest2.partition(":")
        w = float(wstr or 1.0)
        eta = float(etastr or 0.1)
        t0 = time.time()
        torch.manual_seed(a.seed + 17)
        if name == "raw":
            z, zv = zd, zs
        elif name in ("trunc", "trunc_coarse"):
            ch = COARSE if name == "trunc_coarse" else None
            th = float(arg or 0.8)
            z, zv = R.truncate(zd, th, ch), R.truncate(zs, th, ch)
        elif name in ("sir", "sircal"):
            beta = beta_cal if name == "sircal" else 1.0
            k = int(arg or 8)
            gg = torch.Generator().manual_seed(a.seed + 1)
            z = torch.cat([R.sir(G, D, yd[i:i + 64], k, lam, beta, a.n_aug, gg, w) for i in range(0, a.n, 64)])
            zv = R.sir(G, D, ys, k, lam, beta, a.n_aug, gg, w)
        elif name == "trunc_sir":  # candidatos con truncamiento grueso 0.7 + selección por D con guía w
            gg = torch.Generator().manual_seed(a.seed + 1)
            k = int(arg or 8)
            z = torch.cat([R.sir(G, D, yd[i:i + 64], k, lam, beta_cal, a.n_aug, gg, w, 0.7) for i in range(0, a.n, 64)])
            zv = R.sir(G, D, ys, k, lam, beta_cal, a.n_aug, gg, w, 0.7)
        elif name in ("dgflow", "dgflow_coarse"):
            kw = dict(G=G, D=D, lam=lam, steps=int(arg or 25), beta=beta_cal, n_aug=a.n_aug,
                      chunks=COARSE if name == "dgflow_coarse" else None, w=w, eta=eta)
            z, zv = batched(R.dgflow, zd, yd, **kw), batched(R.dgflow, zs, ys, **kw)
        elif name in ("support", "support_trunc"):  # restricción de soporte (≥ min_px de primer plano por vista)
            mp = int(arg or 240)
            tc = 1.0 if name == "support_trunc" else None
            zs0 = R.truncate(zd, tc, COARSE) if tc else zd
            zv0 = R.truncate(zs, tc, COARSE) if tc else zs
            gg = torch.Generator().manual_seed(a.seed + 3)
            z = torch.cat([R.support_resample(G, zs0[i:i + 64], yd[i:i + 64], lam, mp, generator=gg, trunc_coarse=tc)
                           for i in range(0, a.n, 64)])
            zv = R.support_resample(G, zv0, ys, lam, mp, generator=gg, trunc_coarse=tc)
        elif name == "ddls":
            kw = dict(G=G, D=D, lam=lam, steps=int(arg or 200), eps=0.01, beta=beta_cal, n_aug=a.n_aug)
            z, zv = batched(R.ddls, zd, yd, **kw), batched(R.ddls, zs, ys, **kw)
        else:
            raise ValueError(spec)
        x = render(G, z, yd, lam)
        panels[spec] = render(G, zv, ys, lam)
        with torch.no_grad():
            dmean = float(torch.cat([R.d_logit(D, x[i:i + 64].to(dev).float() / 127.5 - 1, yd[i:i + 64], 8)
                                     for i in range(0, a.n, 64)]).mean())
        r = {"d_logit": dmean, "secs": time.time() - t0, "z_norm": float(z.norm(dim=1).mean())}
        for view, sl in (("front", slice(0, 3)), ("back", slice(3, 6))):
            r[f"kid_{view}_x1e3"] = 1e3 * E.kid_from_feats(real[view], E.inception_features(x[:, sl], dev), 1000, 50)[0]
        tm = E.type_metrics(predict_logits(clf, torch.cat([x[:, :3], x[:, 3:]])), torch.cat([y, y]))
        r.update(type_map=tm["mAP"], set_acc=tm["set_acc"])
        xn = x.permute(0, 2, 3, 1).numpy()
        r["coherence"] = float(E.palette_hi_batch(xn[..., :3], xn[..., 3:]).mean())
        r["div_l1"] = float((x[:half].float() - x[half:2 * half].float()).abs().mean() / 255)
        frag = torch.cat([R.count_fragments(x[:, :3]), R.count_fragments(x[:, 3:])]).float()
        r["fragments"] = float(frag.mean())
        r["single_body"] = float((frag == 1).float().mean())
        _, d_gen = E.nearest_neighbors(np.ascontiguousarray(xn[:500, ..., :3]), train_f, k=1, device=dev)
        r["copy_frac"] = float((np.asarray(d_gen)[:, 0] < q05).mean())
        results[spec] = r
        print(f"[{spec}] " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)

    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [f"β calibrado (Platt, especies no vistas vs falsos) = {beta_cal:.3f}", "",
             "| Método | KID×10³ frente ↓ | KID×10³ espalda ↓ | mAP tipo ↑ | Acierto conj. ↑ | Coherencia | "
             "Diversidad L1 | Fragmentos/vista ↓ | % cuerpo único ↑ | Frac. cerca de copia ↓ | logit D | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in results.items():
        if m.startswith("_"):
            continue
        lines.append(f"| {m} | {r['kid_front_x1e3']:.1f} | {r['kid_back_x1e3']:.1f} | {r['type_map']:.3f} | "
                     f"{r['set_acc']:.3f} | {r['coherence']:.3f} | {r['div_l1']:.4f} | {r['fragments']:.2f} | "
                     f"{100 * r['single_body']:.0f} % | {r['copy_frac']:.3f} | {r['d_logit']:+.2f} | {r['secs']:.0f} |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    n_rows, cw, chh = len(ys), 194, 98
    canvas = Image.new("RGB", (cw * len(panels) + 110, chh * n_rows + 24), "white")
    dr = ImageDraw.Draw(canvas)
    for j, (m, p) in enumerate(panels.items()):
        dr.text((110 + j * cw + 40, 4), m, fill=(0, 0, 0), font=font)
        for i in range(n_rows):
            canvas.paste(Image.fromarray(torch.cat([p[i, :3], p[i, 3:]], 2).permute(1, 2, 0).numpy()),
                         (110 + j * cw, 24 + i * chh))
    for i in range(0, n_rows, 4):
        dr.text((4, 24 + i * chh + 40), T.label(ys[i].cpu().numpy()), fill=(0, 0, 0), font=font)
    canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST).save(out / "comparacion_metodos.png")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
