"""Evaluación completa de un checkpoint de PokeGAN -> metrics.json + metrics.md (+ figuras de novedad).

Redes preentrenadas (Inception) SOLO para medir; el clasificador de tipos se entrenó desde cero
(scripts/train_type_classifier.py). Todas las métricas van junto a referencias reales:
techo (datos reales) y piso (barajado), para que los números signifiquen algo.

Uso:
    python scripts/evaluate.py --ckpt runs/main/best.pt --out results/eval --n 5000 --seeds 0 1 2
    # prueba de humo en CPU (minutos):
    python scripts/evaluate.py --ckpt runs/smoke/last.pt --out results/eval_smoke --n 64 --seeds 0 1
        --real-limit 300 --kid-subsets 5 --div-per-type 8 --combo-n 8 --device cpu
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

from pokegan import evaluate as E  # noqa: E402
from pokegan import types as T  # noqa: E402
from pokegan.data import load_curated, to_tensor_pairs  # noqa: E402
from pokegan.sample import load_generator  # noqa: E402
from pokegan.type_classifier import load_classifier, predict_logits  # noqa: E402

# Combinaciones de tipos que NO aparecen en el conjunto de entrenamiento del GAN.
UNSEEN = [("normal", "rock"), ("normal", "bug"), ("normal", "steel"), ("normal", "ice"),
          ("poison", "ice"), ("ground", "fairy"), ("rock", "ghost"), ("bug", "dragon"),
          ("bug", "dark"), ("fire", "fairy"), ("dragon", "fairy")]
TYPE_KEYS = ("mAP", "f1_macro", "set_acc")


def mean_std(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std())}


def views(x_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, 6, H, W] -> (frente [N, 3, H, W], espalda [N, 3, H, W])."""
    return x_u8[:, :3], x_u8[:, 3:]


def hwc(x_u8: torch.Tensor) -> np.ndarray:
    """[N, 3, H, W] uint8 -> numpy [N, H, W, 3]."""
    return x_u8.permute(0, 2, 3, 1).numpy()


def classify_pairs(clf, x_u8: torch.Tensor, y: np.ndarray):
    """Logits de frentes y espaldas como muestras separadas + etiquetas duplicadas."""
    f, b = views(x_u8)
    return predict_logits(clf, torch.cat([f, b])), np.concatenate([y, y])


def monotype_reference(d, per_type: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Filas reales de referencia para diversidad: monotipos no shiny, UN sprite por especie
    (si no, especies con cientos de filas parecerían poco diversas), hasta per_type por tipo."""
    rng = np.random.default_rng(seed)
    mono = d.y.sum(1) == 1
    idx, groups = [], []
    for t in range(T.NUM_TYPES):
        rows = rng.permutation(np.flatnonzero(mono & (d.y[:, t] == 1) & (d.is_shiny == 0)))
        _, first = np.unique(d.species_id[rows], return_index=True)
        pick = rows[np.sort(first)][:per_type]
        idx.append(pick)
        groups.append(np.full(len(pick), t))
    return np.concatenate(idx), np.concatenate(groups)


def frequent_seen_duals(d, k: int) -> list[tuple[str, str]]:
    """Las k combinaciones duales con más especies distintas en entrenamiento (referencia vista)."""
    counts = {}
    for c in T.all_conditions():
        if len(c) == 2:
            i, j = T.TYPE_TO_IDX[c[0]], T.TYPE_TO_IDX[c[1]]
            m = (d.y[:, i] == 1) & (d.y[:, j] == 1)
            counts[c] = len(np.unique(d.species_id[m]))
    return sorted(counts, key=counts.get, reverse=True)[:k]


def run(G, lam, train, val, device, out: Path, n=5000, seeds=(0, 1, 2), clf=None, thresholds=None,
        kid_subsets=100, div_per_type=50, combo_n=200, nn_n=2000) -> dict:
    """Protocolo completo. Devuelve un dict serializable con todas las métricas."""
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"n": n, "seeds": list(seeds), "lam": lam}

    # ---------------- referencias reales
    print("Rasgos Inception de los reales...", flush=True)
    x_tr, x_va = to_tensor_pairs(train, "cpu"), to_tensor_pairs(val, "cpu")
    f_tr = {v: E.inception_features(x, device) for v, x in zip(("front", "back"), views(x_tr))}
    f_va = {v: E.inception_features(x, device) for v, x in zip(("front", "back"), views(x_va))}
    res["real_val_vs_train"] = {
        v: {"kid": E.kid_from_feats(f_tr[v], f_va[v], 1000, kid_subsets)[0],
            "fid": E.fid_from_feats(f_tr[v], f_va[v])} for v in ("front", "back")}
    res["coherence_ref"] = E.coherence_reference(train, n=len(train))
    if clf is not None:
        logits, y2 = classify_pairs(clf, x_va, val.y)
        res["type_ceiling"] = E.type_metrics(logits, y2, thresholds)

    # ---------------- muestras principales (condiciones = distribución de etiquetas de entrenamiento)
    per_seed = []
    x_first = y_first = None
    for s in seeds:
        print(f"Semilla {s}: generando {n} pares...", flush=True)
        rng = np.random.default_rng(s)
        y = train.y[rng.integers(0, len(train), n)]
        x = E.generate_u8(G, torch.from_numpy(y), lam, device, seed=s)
        if x_first is None:
            x_first, y_first = x, y
        r = {}
        for v, xv in zip(("front", "back"), views(x)):
            f = E.inception_features(xv, device)
            r[f"kid_{v}"] = E.kid_from_feats(f_tr[v], f, 1000, kid_subsets, seed=s)[0]
            r[f"fid_{v}"] = E.fid_from_feats(f_tr[v], f)
        f, b = views(x)
        r["coherence"] = float(E.palette_hi_batch(hwc(f), hwc(b)).mean())
        if clf is not None:
            logits, y2 = classify_pairs(clf, x, y)
            r["type_gen"] = E.type_metrics(logits, y2, thresholds)
            perm = rng.permutation(n)  # piso: las mismas imágenes con condiciones barajadas
            r["type_floor"] = E.type_metrics(logits, np.concatenate([y[perm], y[perm]]), thresholds)
        per_seed.append(r)
    res["per_seed"] = per_seed
    for k in ("kid_front", "kid_back", "fid_front", "fid_back", "coherence"):
        res[k] = mean_std([r[k] for r in per_seed])
    if clf is not None:
        for part in ("type_gen", "type_floor"):
            res[part] = {k: mean_std([r[part][k] for r in per_seed]) for k in TYPE_KEYS}
        res["type_gen"]["per_type_recall"] = {
            t: float(np.mean([r["type_gen"]["per_type_recall"].get(t, np.nan) for r in per_seed]))
            for t in T.TYPES}
        ceil, floor, gen = res["type_ceiling"], res["type_floor"], res["type_gen"]
        res["type_normalized"] = {
            k: (gen[k]["mean"] - floor[k]["mean"]) / max(ceil[k] - floor[k]["mean"], 1e-9) for k in TYPE_KEYS}

    # ---------------- diversidad intra-condición: 18 monotipos, hasta div_per_type por tipo
    print("Diversidad intra-condición...", flush=True)
    idx_ref, g_ref = monotype_reference(train, div_per_type)
    x_ref = x_tr[torch.from_numpy(idx_ref)]
    # Vendi crece con el tamaño del grupo (1..n): generamos por tipo tantas muestras como especies
    # reales hay en la referencia (p. ej. solo 5 monotipos Volador), para comparar grupos iguales.
    n_ref = np.bincount(g_ref, minlength=T.NUM_TYPES)
    y_div = np.repeat(np.stack([T.encode([t]) for t in T.TYPES]), n_ref, axis=0)
    g_div = np.repeat(np.arange(T.NUM_TYPES), n_ref)
    x_div = E.generate_u8(G, torch.from_numpy(y_div), lam, device, seed=10_000)

    def pair_feats(x):  # rasgos del par = [Inception(frente); Inception(espalda)]
        return torch.cat([E.inception_features(xv, device) for xv in views(x)], dim=1)

    div_gen = E.diversity(pair_feats(x_div), g_div, x_div.numpy())
    div_real = E.diversity(pair_feats(x_ref), g_ref, x_ref.numpy())
    res["diversity"] = {"gen": div_gen, "real": div_real,
                        "ratio": {k: div_gen[k] / div_real[k] for k in div_gen}}

    # ---------------- novedad: vecino más cercano en píxeles (frentes)
    print("Novedad (vecinos más cercanos)...", flush=True)
    gen_f = hwc(views(x_first[:nn_n])[0])
    idx_g, d_g = E.nearest_neighbors(gen_f, train.front, k=3, device=device)
    _, d_v = E.nearest_neighbors(val.front, train.front, k=1, device=device)
    p5 = float(np.percentile(d_v[:, 0], 5))
    from scipy.stats import mannwhitneyu

    res["novelty"] = {
        "frac_below_val_p5": float((d_g[:, 0] < p5).mean()),
        "val_p5": p5,
        "median_nn_gen": float(np.median(d_g[:, 0])),
        "median_nn_val": float(np.median(d_v[:, 0])),
        "mannwhitney_p_gen_less_than_val": float(mannwhitneyu(d_g[:, 0], d_v[:, 0], alternative="less").pvalue),
    }
    closest = np.argsort(d_g[:, 0])[:12]
    rand = np.random.default_rng(0).choice(len(gen_f), size=min(12, len(gen_f)), replace=False)
    E.nn_figure(gen_f[closest], train.front, idx_g[closest], out / "nn_closest.png", d_g[closest],
                "Novedad: las 12 muestras MÁS parecidas al entrenamiento y sus 3 vecinos (L2 en píxeles)")
    E.nn_figure(gen_f[rand], train.front, idx_g[rand], out / "nn_random.png", d_g[rand],
                "Novedad: 12 muestras al azar y sus 3 vecinos más cercanos del entrenamiento")

    # ---------------- combinaciones nunca vistas vs. vistas frecuentes
    if clf is not None:
        print("Combinaciones no vistas...", flush=True)

        def combo_rate(combos, seed0):
            rates = {}
            for i, c in enumerate(combos):
                y = np.repeat(T.encode(c)[None], combo_n, axis=0)
                x = E.generate_u8(G, torch.from_numpy(y), lam, device, seed=seed0 + i)
                logits, y2 = classify_pairs(clf, x, y)
                rates[T.label(T.encode(c))] = E.type_metrics(logits, y2, thresholds)["set_acc"]
            return rates

        seen = frequent_seen_duals(train, len(UNSEEN))
        unseen_rows = {T.label(T.encode(c)): int(((train.y @ T.encode(c)) == 2).sum()) for c in UNSEEN}
        res["combos"] = {"unseen": combo_rate(UNSEEN, 20_000), "seen": combo_rate(seen, 30_000),
                         "unseen_train_rows": unseen_rows}
        res["combos"]["unseen_mean"] = float(np.mean(list(res["combos"]["unseen"].values())))
        res["combos"]["seen_mean"] = float(np.mean(list(res["combos"]["seen"].values())))
    return res


def fmt(ms: dict, scale: float = 1.0, digits: int = 3) -> str:
    return f"{ms['mean'] * scale:.{digits}f} ± {ms['std'] * scale:.{digits}f}"


def to_markdown(m: dict, ckpt: str, step: int) -> str:
    """Tabla en español con todas las métricas y sus referencias."""
    L = [f"# Evaluación de PokeGAN", "",
         f"Checkpoint `{ckpt}` (paso {step}, λ = {m['lam']:.2f}); {m['n']} pares generados por semilla, "
         f"semillas {', '.join(map(str, m['seeds']))} (media ± desviación entre semillas). "
         "Inception se usa SOLO para evaluar.", "",
         "## Calidad de distribución (Inception-v3, rasgos de 2048 dims)", "",
         "| Métrica | Frente | Espalda |", "|---|---|---|",
         f"| KID ×10³ generado vs. entrenamiento | {fmt(m['kid_front'], 1e3, 2)} | {fmt(m['kid_back'], 1e3, 2)} |",
         f"| FID generado vs. entrenamiento | {fmt(m['fid_front'], 1, 1)} | {fmt(m['fid_back'], 1, 1)} |",
         f"| Referencia: KID ×10³ real (validación) vs. entrenamiento | "
         f"{m['real_val_vs_train']['front']['kid'] * 1e3:.2f} | {m['real_val_vs_train']['back']['kid'] * 1e3:.2f} |",
         f"| Referencia: FID real (validación) vs. entrenamiento* | "
         f"{m['real_val_vs_train']['front']['fid']:.1f} | {m['real_val_vs_train']['back']['fid']:.1f} |", "",
         "\\* El FID está sesgado hacia arriba con N chico (validación ≈ 350 pares): no comparar FIDs con "
         "distinto N. El KID es insesgado y es la métrica titular.", ""]

    if "type_gen" in m:
        g, f, c, nz = m["type_gen"], m["type_floor"], m["type_ceiling"], m["type_normalized"]
        L += [f"## Condicionamiento por tipo ({m.get('clf_desc', 'clasificador entrenado desde cero')}; "
              "protocolo GAN-test)", "",
              "| | mAP | F1 macro | Acierto del conjunto |", "|---|---|---|---|",
              "| Generado | " + " | ".join(fmt(g[k]) for k in TYPE_KEYS) + " |",
              "| Techo: reales de validación | " + " | ".join(f"{c[k]:.3f}" for k in TYPE_KEYS) + " |",
              "| Piso: condiciones barajadas | " + " | ".join(fmt(f[k]) for k in TYPE_KEYS) + " |",
              "| Puntaje normalizado (gen − piso)/(techo − piso) | " + " | ".join(f"{nz[k]:.2f}" for k in TYPE_KEYS)
              + " |", "",
              "Acierto del conjunto = los |c| tipos con mayor puntaje coinciden con la condición. Los umbrales "
              "de F1 se ajustaron en validación (el techo de F1 es algo optimista).", "",
              "| Tipo | Recall generado | Recall real (validación) |", "|---|---|---|"]
        for t in T.TYPES:
            rg = g["per_type_recall"].get(t, float("nan"))
            rc = c["per_type_recall"].get(t, float("nan"))
            L.append(f"| {T.TYPES_ES[t]} | {rg:.3f} | {rc:.3f} |")
        L.append("")
    else:
        L += ["## Condicionamiento por tipo", "",
              "No se encontró `runs/type_classifier/best.pt`: ejecutar `python scripts/train_type_classifier.py`.", ""]

    cr = m["coherence_ref"]
    norm = (m["coherence"]["mean"] - cr["random"]) / max(cr["matched"] - cr["random"], 1e-9)
    L += ["## Coherencia frente/espalda (intersección de histogramas de paleta, 8×8×8 colores)", "",
          "| | HI de paleta |", "|---|---|",
          f"| Generado | {fmt(m['coherence'])} |",
          f"| Real emparejado (techo) | {cr['matched']:.3f} |",
          f"| Real barajado, espalda de otra especie (piso) | {cr['random']:.3f} |",
          f"| Índice normalizado (gen − piso)/(techo − piso) | {norm:.2f} |", ""]

    d = m["diversity"]
    L += ["## Diversidad dentro de una condición (18 monotipos)", "",
          "| | Distancia coseno (Inception, par) | Vendi (criaturas efectivas) | L1 en píxeles |",
          "|---|---|---|---|",
          f"| Generado | {d['gen']['cosine']:.3f} | {d['gen']['vendi']:.1f} | {d['gen']['l1']:.3f} |",
          f"| Real (un sprite por especie) | {d['real']['cosine']:.3f} | {d['real']['vendi']:.1f} | {d['real']['l1']:.3f} |",
          f"| Razón generado / real | {d['ratio']['cosine']:.2f} | {d['ratio']['vendi']:.2f} | {d['ratio']['l1']:.2f} |",
          "", "Por tipo se comparan grupos del mismo tamaño (tantas muestras generadas como especies "
          "reales monotipo, máx. 50), porque Vendi crece con el número de muestras. "
          "Razón cercana a 1 = diversidad como la real; < 0.5 sugiere colapso parcial de modos.", ""]

    nv = m["novelty"]
    L += ["## Novedad (vecino más cercano del entrenamiento, L2 en píxeles, frentes)", "",
          f"- Fracción de generados más cerca del entrenamiento que el 5 % más cercano de los reales de "
          f"validación (especies nunca vistas): **{nv['frac_below_val_p5']:.3f}** (≈ 0.05 si no hay copia).",
          f"- Mediana de la distancia al vecino: generados {nv['median_nn_gen']:.1f} vs. validación "
          f"{nv['median_nn_val']:.1f}; Mann-Whitney (gen < val) p = {nv['mannwhitney_p_gen_less_than_val']:.3g}.",
          "- Figuras: `nn_closest.png` (los más parecidos) y `nn_random.png` (al azar).", ""]

    if "combos" in m:
        cb = m["combos"]
        L += ["## Combinaciones de tipos nunca vistas en entrenamiento", "",
              "Tasa en que AMBOS tipos de la condición quedan en el top-2 del clasificador.", "",
              "| Combinación no vista | Tasa |", "|---|---|"]
        L += [f"| {k} | {v:.3f} |" for k, v in cb["unseen"].items()]
        L += [f"| **Media no vistas** | **{cb['unseen_mean']:.3f}** |",
              f"| Media de {len(cb['seen'])} combinaciones vistas frecuentes (referencia) | {cb['seen_mean']:.3f} |", "",
              f"Filas de entrenamiento con alguna combinación \"no vista\": {sum(cb['unseen_train_rows'].values())} "
              "(debe ser 0).", ""]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "eval"))
    ap.add_argument("--n", type=int, default=5000, help="pares generados por semilla")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default=str(ROOT / "data" / "cache" / "pairs96.npz"))
    ap.add_argument("--clf", default=str(E.CLASSIFIER_PATH))
    ap.add_argument("--kid-subsets", type=int, default=100)
    ap.add_argument("--div-per-type", type=int, default=50, help="máx. de muestras (y de especies reales) por monotipo para diversidad")
    ap.add_argument("--combo-n", type=int, default=200, help="muestras por combinación (vistas/no vistas)")
    ap.add_argument("--real-limit", type=int, default=0,
                    help="solo para pruebas rápidas: usar como máximo este número de pares reales")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    device = torch.device(a.device or ("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"))
    G, info = load_generator(a.ckpt, ema=True, device=device, standing=True)
    train, val = load_curated(a.data)
    if a.real_limit:
        rng = np.random.default_rng(0)
        train = train.subset(np.sort(rng.choice(len(train), min(a.real_limit, len(train)), replace=False)))
        val = val.subset(np.sort(rng.choice(len(val), min(a.real_limit, len(val)), replace=False)))
    clf = thresholds = None
    if Path(a.clf).exists():
        clf, thresholds = load_classifier(a.clf, device)
    else:
        print(f"Aviso: no existe {a.clf}; se omiten las métricas de tipo.", flush=True)

    out = Path(a.out)
    m = run(G, info["lam"], train, val, device, out, n=a.n, seeds=a.seeds, clf=clf, thresholds=thresholds,
            kid_subsets=a.kid_subsets, div_per_type=a.div_per_type, combo_n=a.combo_n, nn_n=min(2000, a.n))
    m.update(ckpt=str(a.ckpt), step=info["step"],
             clf_desc=("linear probe sobre DINOv3 congelado (preentrenado, SOLO evaluación)"
                       if type(clf).__name__ == "TypeProbe" else "clasificador entrenado desde cero"))
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(m, a.ckpt, info["step"])
    (out / "metrics.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nGuardado en {out / 'metrics.json'} y {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
