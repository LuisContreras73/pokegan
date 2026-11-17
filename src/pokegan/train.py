"""Bucle de entrenamiento del GAN condicional de dos vistas.

Un paso = n_dis pasos de D (2 en la config principal) + un paso de G (TTUR: lr_D = 4·lr_G).

Paso de D (G congelado):
    reales  x_r = volteo_conjunto(lote muestreado con pesos)            etiquetas y_r
    falsos  x_f = G(z, y_g, λ_t), y_g = mitad permutada de y_r duplicada (p_g(y) = p_data(y))
    desajustados x_m = (frente_i, espalda_j) de otra especie, misma orientación   etiquetas y_i
    D ve T(·) (DiffAugment pareado) de los tres y minimiza
        H(C_uc) + λ_t·H(C_c) + w_LC·[LeCam(C_uc) + λ_t·LeCam(C_c)]
Paso de G (D congelado):  −E[C_uc(T(G(z,y)))] − λ_t·E[C_c(T(G(z,y)))]  (+ w_ms·MSGAN)
Luego EMA de G.

Las dos mitades del lote de G comparten etiquetas con z distintos ⇒ `div` = mean|G(z1,y) − G(z2,y)|
mide gratis la diversidad dentro de una condición (alerta temprana de colapso de modo).
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import types as T
from .augment import diff_augment, rand_translation
from .data import load_curated, sampling_weights, to_tensor_pairs
from .ema import EMA, standing_stats
from .losses import (LeCam, alpha_shape_penalty, d_hinge, derangement_other_species, g_hinge, mode_seeking,
                     transition_lambda)
from .models import Discriminator, Generator

LOG_COLS = ["step", "time_min", "it_s", "lam", "d_loss", "g_loss", "d_real", "d_fake", "d_mis",
            "r_t", "r_v", "div", "lecam"]
EVAL_COLS = ["kid_front", "kid_back", "type_map", "coherence"]
DUAL_SHOW = [("fire", "flying"), ("water", "ground"), ("grass", "poison"), ("bug", "steel"),
             ("psychic", "fairy"), ("dark", "ghost")]
UNSEEN_SHOW = [("fire", "fairy"), ("dragon", "fairy"), ("rock", "ghost")]


def grid_conditions() -> list[tuple[str, ...]]:
    """Filas de las grillas de monitoreo: 18 monotipos + 6 duales + 3 combinaciones nunca vistas."""
    return [(t,) for t in T.TYPES] + DUAL_SHOW + UNSEEN_SHOW


def to_u8(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def save_grid(x: torch.Tensor, n_cols: int, path: str | Path) -> None:
    """x [filas·n_cols, 6, 96, 96] → PNG; filas = condiciones, columnas = z; celda = frente|espalda."""
    u = to_u8(x).cpu()
    cells = F.pad(torch.cat([u[:, :3], u[:, 3:]], dim=3), [0, 2, 0, 2], value=200)  # N,3,98,194
    rows = cells.shape[0] // n_cols
    h, w = cells.shape[2], cells.shape[3]
    grid = cells.view(rows, n_cols, 3, h, w).permute(0, 3, 1, 4, 2).reshape(rows * h, n_cols * w, 3)
    Image.fromarray(grid.numpy()).save(path)


def _rng_state() -> dict:
    s = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        s["cuda"] = torch.cuda.get_rng_state()
    return s


def _robust(fn, what: str, tries: int = 10) -> bool:
    """Reintenta operaciones de archivo (en Windows fallan si otro proceso tiene el archivo abierto)."""
    for k in range(tries):
        try:
            fn()
            return True
        except OSError as e:
            if k == tries - 1:
                print(f"[aviso] no se pudo {what}: {e}", flush=True)
            else:
                time.sleep(1.0)
    return False


def train(cfg: dict) -> Path:
    tc, lc, dc = cfg["train"], cfg["log"], cfg["data"]
    want = cfg.get("device", "cuda")
    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device(want if (want == "cpu" or has_cuda) else "cpu")
    run_dir = Path(cfg.get("out_dir", "runs")) / cfg["name"]
    bench = tc.get("benchmark")
    if not cfg.get("resume") and not bench and (run_dir / "last.pt").exists():
        raise SystemExit(f"{run_dir} ya tiene last.pt: usa --resume auto o elige otro --name")
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    # Con 24 hilos OpenMP y otros procesos activos, construir el modelo en CPU tardaba minutos
    # (sobresuscripción); unos pocos hilos bastan porque el cómputo pesado va en la GPU.
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.get("tf32", True)
        torch.backends.cudnn.allow_tf32 = cfg.get("tf32", True)
        torch.backends.cudnn.benchmark = True
        if cfg.get("memory_fraction"):
            torch.cuda.set_per_process_memory_fraction(cfg["memory_fraction"])

    # ---------------- datos (todo en el dispositivo como uint8) ----------------
    tr, va = load_curated(dc["cache"], dc["val_frac"], cfg["seed"])
    if dc.get("limit"):
        tr = tr.subset(np.arange(len(tr)) < dc["limit"])
        va = va.subset(np.arange(len(va)) < max(8, dc["limit"] // 8))
    X = to_tensor_pairs(tr, device)
    Y = torch.from_numpy(tr.y).to(device)
    S = torch.from_numpy(tr.species_id.astype(np.int64)).to(device)
    Wt = torch.from_numpy(sampling_weights(tr, dc["species_power"], dc["combo_power"], dc["shiny_factor"]))
    Wt = Wt.float().to(device)
    Xv = to_tensor_pairs(va, device)[:256].float().div(127.5).sub(1)
    Yv = torch.from_numpy(va.y[:256]).to(device)
    label_bank = torch.from_numpy(tr.y).to(torch.uint8)  # filas uniformes (misma mezcla que la referencia de KID)
    # etiquetas con la MISMA mezcla que vio la BN de G durante el entrenamiento (muestreo ponderado)
    bn_bank = Y[torch.multinomial(Wt, 20000, replacement=True)].to(torch.uint8).cpu()
    bn_dev = bn_bank.to(device).float()
    print(f"[datos] train {len(tr)} pares ({len(np.unique(tr.species_id))} especies) | "
          f"val {len(va)} pares | dispositivo {device}")

    # ---------------- modelos, optimizadores, reanudación ----------------
    G = Generator(**cfg["G"]).to(device)
    D = Discriminator(**cfg["D"]).to(device)
    betas = tuple(tc["betas"])
    zparams = [p for n, p in G.named_parameters() if n.startswith("zmap")]
    base = [p for n, p in G.named_parameters() if not n.startswith("zmap")]
    groups = [{"params": base}] + ([{"params": zparams, "lr": tc["lr_g"] * tc.get("zmap_lr_mult", 10.0)}] if zparams else [])
    optG = torch.optim.Adam(groups, lr=tc["lr_g"], betas=betas, eps=tc["eps"])
    optD = torch.optim.Adam(D.parameters(), lr=tc["lr_d"], betas=betas, eps=tc["eps"])
    ema = EMA(G, tc["ema_decay"], tc["ema_start"])
    mcfg = tc.get("multiscale")  # D auxiliares desde cero: par a 48/24 px + siluetas (ver models/multiscale_d.py)
    MS = optM = None
    if mcfg:
        from .models.multiscale_d import MultiScaleD

        MS = MultiScaleD(mcfg.get("ch", cfg["D"].get("ch", 32)), tuple(mcfg.get("scales", (48, 24))),
                         mcfg.get("mask", True)).to(device)
        optM = torch.optim.Adam(MS.parameters(), lr=tc["lr_d"], betas=betas, eps=tc["eps"])
        print(f"[multiescala] escalas {MS.scales} + siluetas={MS.mask_d is not None} | "
              f"{sum(p.numel() for p in MS.parameters()) / 1e6:.2f} M parámetros", flush=True)
    vcfg = tc.get("vision_d")  # discriminador asistido por visión (preentrenado congelado), opcional
    VD = optV = None
    if vcfg:
        from .models.vision_d import VisionD

        VD = VisionD(vcfg.get("backbone", "convnext_tiny.fb_in22k_ft_in1k"), vcfg.get("res", 160)).to(device)
        optV = torch.optim.Adam(VD.trainable_parameters(), lr=vcfg.get("lr", 2e-4), betas=betas, eps=tc["eps"])
        print(f"[vision-D] backbone {vcfg.get('backbone', 'convnext_tiny.fb_in22k_ft_in1k')} congelado | "
              f"peso en G {vcfg.get('weight', 1.0)}", flush=True)
    lecam_uc, lecam_cc = LeCam(tc["lecam_decay"]), LeCam(tc["lecam_decay"])
    total = tc["iters"]
    t_start, t_end = int(tc["t_start_frac"] * total), int(tc["t_end_frac"] * total)
    step, best = 0, math.inf
    if cfg.get("resume"):
        # map_location='cpu': los estados RNG deben seguir siendo ByteTensor de CPU; load_state_dict copia
        # los pesos al dispositivo y Adam mueve sus momentos al dispositivo de los parámetros.
        ck = torch.load(cfg["resume"], map_location="cpu", weights_only=False)
        strict = not cfg["G"].get("spatial_z", False)  # al activar la ruta espacial, sus pesos nuevos no existen
        G.load_state_dict(ck["G"], strict=strict)
        ema.module.load_state_dict(ck["G_ema"], strict=strict)
        D.load_state_dict(ck["D"])
        try:
            optG.load_state_dict(ck["optG"])
        except ValueError:
            print("[resume] optimizador de G reiniciado (cambiaron los grupos de parámetros)", flush=True)
        optD.load_state_dict(ck["optD"])
        for opt, lr in ((optG, tc["lr_g"]), (optD, tc["lr_d"])):  # la config manda (respaldo: bajar lr)
            for gi, g in enumerate(opt.param_groups):
                mult = tc.get("zmap_lr_mult", 10.0) if (opt is optG and gi == 1) else 1.0
                g.update(lr=lr * mult, betas=betas, eps=tc["eps"])
        if MS is not None and "MS" in ck:
            MS.load_state_dict(ck["MS"])
            optM.load_state_dict(ck["optM"])
        lc_state = ck["lecam"]
        if "uc" in lc_state:
            lecam_uc.load_state_dict(lc_state["uc"])
            lecam_cc.load_state_dict(lc_state["cc"])
        lecam_uc.decay = lecam_cc.decay = tc["lecam_decay"]
        step, best = ck["step"], ck.get("best", math.inf)
        if "sched" in ck:  # el calendario de λ_t se fija en la primera corrida; --iters solo mueve el final
            t_start, t_end = ck["sched"]["t_start"], ck["sched"]["t_end"]
        if (run_dir / "best.pt").exists():
            try:
                best = min(best, torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["best"])
            except Exception:
                pass
        torch.set_rng_state(ck["rng"]["cpu"])
        if device.type == "cuda" and "cuda" in ck["rng"]:
            torch.cuda.set_rng_state(ck["rng"]["cuda"])
        lam_now = transition_lambda(step, t_start, t_end)
        if abs(lam_now - ck.get("lam", lam_now)) > 1e-3:
            print(f"[aviso] λ guardado {ck['lam']:.3f} ≠ λ actual {lam_now:.3f}")
        print(f"[resume] desde {cfg['resume']} en el paso {step} | lr_G {tc['lr_g']} lr_D {tc['lr_d']} "
              f"| ms_weight {tc['ms_weight']}")
    print(f"[modelo] G {sum(p.numel() for p in G.parameters()) / 1e6:.2f} M | "
          f"D {sum(p.numel() for p in D.parameters()) / 1e6:.2f} M parámetros")

    # z fijos para las grillas (mismo z por columna ⇒ se ve el efecto del tipo con z fijo)
    conds = grid_conditions()
    n_z = lc.get("grid_z", 4)
    zc = torch.randn(n_z, G.z_dim, generator=torch.Generator().manual_seed(1234))
    gz = zc.repeat(len(conds), 1).to(device)
    gy = torch.from_numpy(np.stack([T.encode(list(c)) for c in conds])).repeat_interleave(n_z, 0).to(device)
    (run_dir / "samples" / "conditions.txt").write_text(
        "\n".join(T.label(T.encode(list(c))) for c in conds), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    if bench:
        total = step + bench
    B, half = tc["batch"], tc["batch"] // 2
    max_sec = tc.get("max_hours") and tc["max_hours"] * 3600
    aug = tc["aug"]
    flip_p = dc["flip_p"]
    lam = transition_lambda(step, t_start, t_end)

    log_path = run_dir / "log.csv"
    new_log = not log_path.exists() or log_path.stat().st_size == 0
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_f, LOG_COLS + EVAL_COLS)
    if new_log and not bench:
        writer.writeheader()
        log_f.flush()

    def ckpt_obj(full: bool = True) -> dict:
        obj = {"G_ema": ema.module.state_dict(), "step": step, "lam": lam, "cfg": cfg,
               "label_bank": label_bank, "bn_label_bank": bn_bank, "best": best,
               "sched": {"t_start": t_start, "t_end": t_end}}
        if full:
            obj.update(G=G.state_dict(), D=D.state_dict(), optG=optG.state_dict(), optD=optD.state_dict(),
                       lecam={"uc": lecam_uc.state_dict(), "cc": lecam_cc.state_dict()}, rng=_rng_state())
            if MS is not None:
                obj.update(MS=MS.state_dict(), optM=optM.state_dict())
        return obj

    def checkpoint(path: Path, full: bool = True) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt_obj(full), tmp)
        if not _robust(lambda: os.replace(tmp, path), f"escribir {path.name}"):
            print(f"[aviso] el checkpoint quedó en {tmp}", flush=True)

    @torch.no_grad()
    def sample_grid() -> None:
        standing_stats(ema.module, bn_dev, n_batches=16, batch=64, lam=lam)
        x = torch.cat([ema.module(gz[i:i + 64], gy[i:i + 64], lam) for i in range(0, len(gz), 64)])
        save_grid(x, n_z, run_dir / "samples" / f"step_{step:06d}.png")

    acc: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=device))
    n_acc, r_v = 0, float("nan")
    t0 = t_log = time.time()
    s_log = step
    warm_t, warm_s = None, None
    warned_budget = False
    print(f"[train] pasos {step}→{total} | λ: 0 hasta {t_start}, 1 desde {t_end} | lote {B}")
    try:
        while step < total:
            lam = transition_lambda(step, t_start, t_end)
            for _ in range(tc.get("n_dis", 1)):  # n_dis pasos de D por cada paso de G
                idx = torch.multinomial(Wt, B, replacement=True)
                x0 = X[idx].float().div_(127.5).sub_(1)
                flip = (torch.rand(B, device=device) < flip_p)[:, None, None, None]
                xr = torch.where(flip, x0.flip(-1), x0)  # volteo conjunto de ambas vistas (solo reales)
                yr, sr = Y[idx], S[idx]
                yh = yr[torch.randperm(B, device=device)[:half]]
                yg = torch.cat([yh, yh])

                # ---------------- paso de D ----------------
                z = torch.randn(B, G.z_dim, device=device)
                with torch.no_grad():
                    xf = G(z, yg, lam)
                # desajustados: frente_i + espalda_j (otra especie) con UNA orientación para todo el par
                # (si cada vista trajera su propio volteo, D los rechazaría por orientación, no por identidad)
                j, valid = derangement_other_species(sr[:half])
                xm = torch.cat([x0[:half, :3], x0[:half][j, 3:]], 1)
                xm = torch.where(flip[:half], xm.flip(-1), xm)
                uc, pj = D(diff_augment(torch.cat([xr, xf, xm]), **aug), torch.cat([yr, yg, yr[:half]]))
                cc = uc + pj
                uc_r, uc_f, uc_m = uc[:B], uc[B:2 * B], uc[2 * B:]
                cc_r, cc_f, cc_m = cc[:B], cc[B:2 * B], cc[2 * B:]
                mw = tc["mis_weight"]
                fw = tc.get("fake_weight", 1.0)
                pair = tc.get("mis_mode", "absolute") == "pair"  # desajustado i comparte frente con real i
                loss_d = (d_hinge(uc_r, uc_f, uc_m, mw, valid, fw, uc_r[:half] if pair else None)
                          + lam * d_hinge(cc_r, cc_f, cc_m, mw, valid, fw, cc_r[:half] if pair else None))
                lc_pen = torch.zeros((), device=device)
                if tc["lecam_weight"] > 0 and step >= tc["lecam_start"]:
                    # LeCam sobre los MISMOS logits que entrena la hinge y con el mismo peso λ_t
                    lc_pen = lecam_uc.penalty(uc_r, uc_f) + lam * lecam_cc.penalty(cc_r, cc_f)
                    loss_d = loss_d + tc["lecam_weight"] * lc_pen
                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()
                lecam_uc.update(uc_r, uc_f)
                lecam_cc.update(cc_r, cc_f)
                if MS is not None:
                    xrf = torch.cat([xr, xf])
                    outs = MS(diff_augment(xrf, **aug), torch.cat([yr, yg]), rand_translation(xrf, aug["translation"]))
                    loss_m = sum(d_hinge(u[:B], u[B:]) + lam * d_hinge((u + q)[:B], (u + q)[B:]) for u, q in outs)
                    optM.zero_grad(set_to_none=True)
                    loss_m.backward()
                    optM.step()
                if VD is not None:
                    vr = VD(diff_augment(xr, **aug), yr)
                    vf = VD(diff_augment(xf, **aug), yg)
                    loss_v = d_hinge(vr, vf)
                    optV.zero_grad(set_to_none=True)
                    loss_v.backward()
                    optV.step()

            # ---------------- paso de G ----------------
            D.requires_grad_(False)
            z = torch.randn(B, G.z_dim, device=device)
            w_area, w_tv = tc.get("alpha_area_weight", 0.0), tc.get("alpha_tv_weight", 0.0)
            if w_area > 0 or w_tv > 0:
                xg, alpha = G(z, yg, lam, return_alpha=True)
            else:
                xg = G(z, yg, lam)
            ug, pg = D(diff_augment(xg, **aug), yg)
            loss_g = g_hinge(ug) + lam * g_hinge(ug + pg)
            if w_area > 0 or w_tv > 0:
                p_area, p_tv = alpha_shape_penalty(alpha, xg, tc.get("alpha_min_area", 240.0),
                                                   tc.get("alpha_tv_target", 0.04))
                loss_g = loss_g + w_area * p_area + w_tv * p_tv
            if MS is not None:
                MS.requires_grad_(False)
                outs = MS(diff_augment(xg, **aug), yg, rand_translation(xg, aug["translation"]))
                loss_g = loss_g + mcfg.get("weight", 1.0) * sum(g_hinge(u) + lam * g_hinge(u + q) for u, q in outs)
                MS.requires_grad_(True)
            if VD is not None:
                VD.requires_grad_(False)
                loss_g = loss_g + vcfg.get("weight", 1.0) * g_hinge(VD(diff_augment(xg, **aug), yg))
                for p_ in VD.trainable_parameters():
                    p_.requires_grad_(True)
            div = (xg[:half] - xg[half:]).abs().mean()
            if tc["ms_weight"] > 0:
                loss_g = loss_g + tc["ms_weight"] * mode_seeking(xg[:half], xg[half:], z[:half], z[half:])
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()
            D.requires_grad_(True)
            step += 1
            ema.update(G, step)

            # ---------------- registro ----------------
            with torch.no_grad():
                eff = uc.detach() + lam * pj.detach()
                vm = valid.float()
                acc["d_loss"] += loss_d.detach()
                acc["g_loss"] += loss_g.detach()
                acc["d_real"] += eff[:B].mean()
                acc["d_fake"] += eff[B:2 * B].mean()
                acc["d_mis"] += (eff[2 * B:] * vm).sum() / vm.sum().clamp_min(1.0)
                acc["r_t"] += eff[:B].sign().mean()
                acc["div"] += div.detach()
                acc["lecam"] += lc_pen.detach()
                n_acc += 1

            if bench:
                if step - (total - bench) == min(20, bench // 2):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    warm_t, warm_s = time.time(), step
                continue

            lam = transition_lambda(step, t_start, t_end)  # λ vigente tras el paso (muestreo/checkpoint)

            if step % lc["sample_every"] == 0:
                with torch.no_grad():
                    uv, pv = D(diff_augment(Xv, **aug), Yv)
                    r_v = (uv + lam * pv).sign().mean().item()
                sample_grid()

            row = None
            if step % lc["log_every"] == 0:
                now = time.time()
                row = {k: (acc[k] / n_acc).item() for k in ["d_loss", "g_loss", "d_real", "d_fake", "d_mis",
                                                             "r_t", "div", "lecam"]}
                if not all(math.isfinite(v) for v in row.values()):
                    checkpoint(run_dir / "nan_abort.pt")
                    raise FloatingPointError(f"Pérdida no finita en el paso {step}: {row}")
                it_s = (step - s_log) / max(now - t_log, 1e-9)
                eta_h = (total - step) / max(it_s, 1e-9) / 3600
                row.update(step=step, time_min=(now - t0) / 60, it_s=it_s, lam=lam, r_v=r_v)
                print(f"[{step:6d}/{total}] λ={lam:.2f} D={row['d_loss']:.3f} G={row['g_loss']:.3f} "
                      f"C(real)={row['d_real']:+.2f} C(fake)={row['d_fake']:+.2f} C(mis)={row['d_mis']:+.2f} "
                      f"r_t={row['r_t']:.2f} r_v={r_v:.2f} div={row['div']:.3f} | {it_s:.2f} it/s, "
                      f"ETA {eta_h:.2f} h", flush=True)
                if max_sec and not warned_budget and lam < 1.0:
                    need = (t_end - step) / max(it_s, 1e-9)
                    if need > max_sec - (now - t0):
                        print(f"[aviso] a {it_s:.2f} it/s, λ_t no llega a 1 antes de --max-hours "
                              f"(faltan {need / 3600:.2f} h): conviene bajar --iters", flush=True)
                    warned_budget = True
                acc.clear()
                n_acc, t_log, s_log = 0, now, step

            if lc.get("eval_every") and step % lc["eval_every"] == 0:
                try:
                    from .evaluate import quick_eval
                    standing_stats(ema.module, bn_dev, n_batches=16, batch=64, lam=lam)
                    res = quick_eval(ema.module, lam, label_bank, device, n=lc.get("eval_n", 2000))
                    print(f"[eval {step}] " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
                    row = row or {"step": step, "lam": lam}
                    row.update({k: res[k] for k in EVAL_COLS if k in res})
                    checkpoint(run_dir / f"snap_{step:06d}.pt", full=False)  # solo G_ema (~35 MB)
                    score = (res["kid_front"] + res["kid_back"]) / 2
                    if lam >= 1.0 and score < best:
                        best = score
                        checkpoint(run_dir / "best.pt", full=False)
                        print(f"[eval {step}] nuevo mejor KID medio {best:.4f} → best.pt", flush=True)
                except Exception as e:  # la evaluación nunca debe tumbar el entrenamiento
                    print(f"[eval {step}] omitida: {type(e).__name__}: {e}", flush=True)

            if row is not None:
                writer.writerow(row)
                log_f.flush()

            if step % lc["ckpt_every"] == 0:
                checkpoint(run_dir / f"ckpt_{step:06d}.pt")
                checkpoint(run_dir / "last.pt")
                for p in sorted(run_dir.glob("ckpt_*.pt"))[:-lc.get("keep_ckpts", 3)]:
                    _robust(lambda p=p: p.unlink(missing_ok=True), f"borrar {p.name}", tries=2)

            if max_sec and time.time() - t0 > max_sec:
                print(f"[train] límite de tiempo alcanzado en el paso {step} (λ={lam:.2f})")
                break
    except KeyboardInterrupt:
        print(f"\n[train] interrumpido en el paso {step}; guardando last.pt")
    except Exception as e:  # OOM, error CUDA, etc.: guardar lo posible antes de salir
        print(f"\n[train] error en el paso {step}: {type(e).__name__}: {e}", flush=True)
        if not bench:
            try:
                checkpoint(run_dir / "crash.pt")
                print(f"[train] estado guardado en {run_dir / 'crash.pt'} (reanudar con --resume)", flush=True)
            except Exception as e2:
                print(f"[train] tampoco se pudo guardar: {e2}", flush=True)
        raise
    finally:
        log_f.close()

    if bench:
        if device.type == "cuda":
            torch.cuda.synchronize()
        its = (step - warm_s) / (time.time() - warm_t) if warm_t else float("nan")
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"[benchmark] {its:.2f} it/s (lote {B}) | memoria pico {mem:.2f} GiB")
        (run_dir / "benchmark.json").write_text(json.dumps({"it_s": its, "mem_gib": mem, "batch": B}))
        return run_dir

    lam = transition_lambda(step, t_start, t_end)
    checkpoint(run_dir / "last.pt")
    sample_grid()
    print(f"[train] fin en el paso {step}; checkpoint en {run_dir / 'last.pt'}")
    return run_dir
