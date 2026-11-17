"""Métricas de evaluación del GAN (redes preentrenadas SOLO aquí, nunca en G/D ni en las pérdidas).

Bloques:
  1. Tipo (GAN-test): mAP / F1 / set-accuracy de un clasificador entrenado desde cero en reales.
  2. Coherencia frente/espalda: intersección de histogramas de paleta (sin entrenamiento).
  3. Distribución: KID (titular) y FID con rasgos Inception-v3 de 2048 dimensiones.
  4. Diversidad intra-condición y novedad (vecinos más cercanos en píxeles).
  5. quick_eval: versión rápida para llamar durante el entrenamiento.

Convención de imágenes: uint8, blanco = 255. Tensores [N, 3, 96, 96] para Inception y el
clasificador; arrays [N, 96, 96, 3] para paleta y vecinos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .types import NUM_TYPES, TYPES

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "cache" / "pairs96.npz"
REAL_FEATS_PATH = ROOT / "data" / "cache" / "eval_real_feats.pt"
CLASSIFIER_PATH = ROOT / "runs" / "type_classifier" / "best.pt"


# ---------------------------------------------------------------------------------------------
# 1. Métricas de tipo
# ---------------------------------------------------------------------------------------------

def type_metrics(logits_or_probs, y_true, thresholds=None) -> dict:
    """Métricas multi-etiqueta de tipo.

    logits_or_probs: [N, 18]. Si hay valores fuera de [0, 1] se interpretan como logits (sigmoide).
    y_true: multi-hot [N, 18]. thresholds: umbral por clase (por defecto 0.5) para F1 y recall.

    Devuelve:
      mAP            media de average precision por clase (solo clases presentes en y_true).
      f1_macro       F1 macro con los umbrales (solo clases presentes).
      set_acc        fracción donde los |c| tipos con mayor puntaje == el conjunto verdadero c.
                     Para combos duales equivale a "ambos tipos en el top-2".
      per_type_recall  {tipo: recall} con los umbrales (solo clases presentes).
    """
    from sklearn.metrics import average_precision_score, f1_score

    s = torch.as_tensor(logits_or_probs).float().cpu()
    if s.numel() and (s.min() < 0 or s.max() > 1):
        s = torch.sigmoid(s)
    p = s.numpy()
    y = (np.asarray(torch.as_tensor(y_true).float().cpu()) > 0.5)
    th = np.full(NUM_TYPES, 0.5) if thresholds is None else np.asarray(thresholds)
    pred = p >= th[None, :]

    present = np.flatnonzero(y.any(0))
    ap = [average_precision_score(y[:, t], p[:, t]) for t in present]
    f1 = f1_score(y, pred, labels=present, average="macro", zero_division=0)

    k = y.sum(1)
    order = np.argsort(-p, axis=1)
    hits = [set(order[i, :k[i]]) == set(np.flatnonzero(y[i])) for i in range(len(y))]

    recall = {TYPES[t]: float(pred[y[:, t], t].mean()) for t in present}
    return dict(mAP=float(np.mean(ap)) if ap else float("nan"), f1_macro=float(f1),
                set_acc=float(np.mean(hits)) if hits else float("nan"), per_type_recall=recall)


# ---------------------------------------------------------------------------------------------
# 2. Coherencia de paleta frente/espalda
# ---------------------------------------------------------------------------------------------

def _color_histograms(imgs: np.ndarray, bins: int, white_min: int) -> np.ndarray:
    """Histogramas de color [N, bins^3] de los píxeles NO blancos, normalizados a suma 1.
    Una imagen sin primer plano deja una fila de ceros."""
    n = len(imgs)
    q = (imgs.reshape(n, -1, 3).astype(np.int64) * bins) // 256          # nivel por canal
    code = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]           # N, P en [0, bins^3)
    fg = imgs.reshape(n, -1, 3).min(-1) < white_min                    # primer plano
    code = code + (np.arange(n) * bins**3)[:, None]                    # desplaza por imagen
    h = np.bincount(code[fg], minlength=n * bins**3).reshape(n, bins**3).astype(np.float64)
    return h / np.maximum(h.sum(1, keepdims=True), 1)


def palette_hi_batch(fronts: np.ndarray, backs: np.ndarray, bins: int = 8, white_min: int = 245,
                     chunk: int = 512) -> np.ndarray:
    """HI_i = Σ_b min(h_frente_i[b], h_espalda_i[b]) con histogramas de 8x8x8 colores.

    1 = misma distribución de colores, 0 = paletas disjuntas. Si una de las vistas es toda blanca
    (sin primer plano) el resultado es 0: no hay paleta que compartir (un G que borra una vista
    debe ser penalizado, no ignorado). fronts/backs: uint8 [N, H, W, 3].
    """
    out = []
    for i in range(0, len(fronts), chunk):
        hf = _color_histograms(np.asarray(fronts[i:i + chunk]), bins, white_min)
        hb = _color_histograms(np.asarray(backs[i:i + chunk]), bins, white_min)
        out.append(np.minimum(hf, hb).sum(1))
    return np.concatenate(out) if out else np.zeros(0)


def palette_hi(front_u8: np.ndarray, back_u8: np.ndarray, bins: int = 8, white_min: int = 245) -> float:
    """Intersección de histogramas de paleta de UN par (ver palette_hi_batch). Imágenes [H, W, 3]."""
    return float(palette_hi_batch(front_u8[None], back_u8[None], bins, white_min)[0])


def shuffled_other_species(species_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutación j(i) tal que, cuando es posible, la especie de j difiere de la de i."""
    j = rng.permutation(len(species_id))
    for _ in range(100):
        bad = np.flatnonzero(species_id[j] == species_id)
        if len(bad) == 0:
            break
        j[bad] = rng.integers(0, len(species_id), len(bad))
    return j


def coherence_reference(d, n: int = 2000, seed: int = 0) -> dict:
    """Calibración en datos reales (PairData): HI medio de pares verdaderos ("matched") y de
    frentes con espaldas de OTRA especie ("random"). La investigación midió 0.706 vs 0.139."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d), size=min(n, len(d)), replace=False)
    j = idx[shuffled_other_species(d.species_id[idx], rng)]
    matched = palette_hi_batch(d.front[idx], d.back[idx])
    random = palette_hi_batch(d.front[idx], d.back[j])
    return dict(matched=float(matched.mean()), random=float(random.mean()), n=int(len(idx)))


# ---------------------------------------------------------------------------------------------
# 3. KID y FID con rasgos Inception-v3
# ---------------------------------------------------------------------------------------------
# Extractor: la Inception-v3 estándar de FID de torch-fidelity (la misma que usa torchmetrics por
# dentro): entrada uint8, la redimensiona a 299 px (bilineal estilo TF1), rasgos pool3 de 2048 dims.
# KID y FID se calculan aquí a partir de los rasgos, así los rasgos reales se guardan en caché y se
# reutilizan entre semillas y durante el entrenamiento. No usamos torchmetrics directamente porque
# su import arrastra `transformers` (20-50 s). tests/test_evaluate.py comprueba que nuestro KID
# coincide con el de torch-fidelity.

_INCEPTION: dict[str, torch.nn.Module] = {}


def _inception(device) -> torch.nn.Module:
    key = str(device)
    if key not in _INCEPTION:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        net = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"])
        _INCEPTION[key] = net.to(device).eval()
    return _INCEPTION[key]


@torch.no_grad()
def inception_features(x_u8: torch.Tensor, device=None, batch: int = 128) -> torch.Tensor:
    """Rasgos Inception [N, 2048] (float64, CPU) de imágenes uint8 [N, 3, H, W]."""
    device = torch.device(device or ("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"))
    net = _inception(device)
    feats = [net(x_u8[i:i + batch].to(device))[0].double().cpu() for i in range(0, len(x_u8), batch)]
    return torch.cat(feats)


def kid_from_feats(f_real: torch.Tensor, f_fake: torch.Tensor, subset_size: int = 1000,
                   subsets: int = 50, seed: int = 0) -> tuple[float, float]:
    """KID = MMD² insesgado con kernel polinómico cúbico k(x, y) = (xᵀy / d + 1)³.

    En cada uno de `subsets` subconjuntos aleatorios de tamaño m (real X, generado Y):
        MMD² = Σ_{i≠j} k(x_i, x_j) / (m(m-1)) + Σ_{i≠j} k(y_i, y_j) / (m(m-1)) - 2 Σ_{i,j} k(x_i, y_j) / m²
    Devuelve (media, desviación) sobre subconjuntos. m = min(subset_size, n_real, n_fake).
    Insesgado ⇒ puede dar valores levemente negativos cuando las distribuciones coinciden.
    """
    f_real, f_fake = f_real.double(), f_fake.double()
    m = min(subset_size, len(f_real), len(f_fake))
    d = f_real.shape[1]
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(subsets):
        x = f_real[torch.randperm(len(f_real), generator=g)[:m]]
        y = f_fake[torch.randperm(len(f_fake), generator=g)[:m]]
        k_xx = (x @ x.T / d + 1) ** 3
        k_yy = (y @ y.T / d + 1) ** 3
        k_xy = (x @ y.T / d + 1) ** 3
        mmd = ((k_xx.sum() - k_xx.diagonal().sum()) / (m * (m - 1))
               + (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
               - 2 * k_xy.sum() / m**2)
        vals.append(float(mmd))
    return float(np.mean(vals)), float(np.std(vals))


def fid_from_feats(f_real: torch.Tensor, f_fake: torch.Tensor) -> float:
    """FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 (Σ_r Σ_g)^{1/2}) en float64.

    Tr((Σ_r Σ_g)^{1/2}) = Σ sqrt(autovalores de Σ_r Σ_g) (mismo cálculo que torchmetrics).
    Ojo: sesgado hacia arriba para N finito; solo comparar FIDs con el mismo N.
    """
    a, b = f_real.double(), f_fake.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = torch.cov(a.T), torch.cov(b.T)
    tr_sqrt = torch.linalg.eigvals(s_a @ s_b).sqrt().real.sum()
    return float((mu_a - mu_b).square().sum() + s_a.trace() + s_b.trace() - 2 * tr_sqrt)


def kid_fid(real_u8: torch.Tensor, fake_u8: torch.Tensor, device=None, fid: bool = True,
            subset_size: int = 1000, subsets: int = 50) -> dict:
    """KID (media, desv.) y FID entre dos conjuntos de imágenes uint8 [N, 3, 96, 96]."""
    f_r = inception_features(real_u8, device)
    f_g = inception_features(fake_u8, device)
    kid, kid_std = kid_from_feats(f_r, f_g, subset_size, subsets)
    out = dict(kid=kid, kid_std=kid_std)
    if fid:
        out["fid"] = fid_from_feats(f_r, f_g)
    return out


# ---------------------------------------------------------------------------------------------
# 4. Diversidad y novedad
# ---------------------------------------------------------------------------------------------

def mean_pairwise_cosine_distance(feats: torch.Tensor) -> float:
    """1 - similitud coseno media entre pares distintos (i ≠ j). 0 = todas iguales."""
    f = torch.nn.functional.normalize(torch.as_tensor(feats).double(), dim=1)
    n = len(f)
    sim = f @ f.T
    return float(1 - (sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))


def vendi_score(feats: torch.Tensor) -> float:
    """Vendi = exp(entropía de los autovalores de K/n), K = matriz de similitud coseno.
    Se lee como "número efectivo de criaturas distintas" (1 .. n)."""
    f = torch.nn.functional.normalize(torch.as_tensor(feats).double(), dim=1)
    lam = torch.linalg.eigvalsh(f @ f.T / len(f)).clamp_min(0)
    lam = lam[lam > 1e-12]
    return float(torch.exp(-(lam * lam.log()).sum()))


def mean_pairwise_l1(imgs_u8) -> float:
    """Distancia L1 media por píxel (escala [0, 1]) entre pares distintos de imágenes."""
    x = torch.as_tensor(np.asarray(imgs_u8)).reshape(len(imgs_u8), -1).float() / 255
    n = len(x)
    d = torch.cdist(x, x, p=1) / x.shape[1]
    return float(d.sum() / (n * (n - 1)))


def diversity(feats: torch.Tensor, groups=None, imgs_u8=None) -> dict:
    """Diversidad intra-condición promediada sobre grupos (p. ej. un grupo por tipo).

    feats: rasgos [N, d] (p. ej. Inception). groups: id de condición por fila (None = un grupo).
    imgs_u8: opcional, imágenes [N, ...] para la L1 en píxeles.
    Devuelve medias sobre grupos de: cosine (distancia coseno media), vendi y, si hay imágenes, l1.
    """
    feats = torch.as_tensor(feats)
    groups = np.zeros(len(feats), int) if groups is None else np.asarray(groups)
    res = {"cosine": [], "vendi": [], "l1": []}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        res["cosine"].append(mean_pairwise_cosine_distance(feats[torch.from_numpy(m)]))
        res["vendi"].append(vendi_score(feats[torch.from_numpy(m)]))
        if imgs_u8 is not None:
            res["l1"].append(mean_pairwise_l1(np.asarray(imgs_u8)[m]))
    return {k: float(np.mean(v)) for k, v in res.items() if v}


@torch.no_grad()
def nearest_neighbors(gen_u8, train_u8, k: int = 3, device=None, gen_chunk: int = 256,
                      train_chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """k vecinos más cercanos de cada imagen generada entre las de entrenamiento, en L2 de píxeles
    (escala [0, 1]). Entradas uint8 [M, ...] y [N, ...] con la misma forma por imagen.

    Recorre en bloques para no llenar la memoria; las distancias finales se recalculan de forma
    exacta (una copia exacta da 0).
    Devuelve (idx int64 [M, k], dist float32 [M, k]) ordenados de menor a mayor distancia.
    """
    device = torch.device(device or ("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"))
    gen = torch.as_tensor(np.asarray(gen_u8)).reshape(len(gen_u8), -1)
    train = torch.as_tensor(np.asarray(train_u8)).reshape(len(train_u8), -1).to(device)
    k = min(k, len(train))
    all_idx, all_dist = [], []
    for s in range(0, len(gen), gen_chunk):
        g = gen[s:s + gen_chunk].to(device).float() / 255
        best_d = torch.full((len(g), k), float("inf"), device=device)
        best_i = torch.zeros((len(g), k), dtype=torch.long, device=device)
        for t0 in range(0, len(train), train_chunk):
            t = train[t0:t0 + train_chunk].float() / 255
            dist = torch.cdist(g, t)
            ids = torch.arange(t0, t0 + len(t), device=device).expand(len(g), -1)
            cand_d, cand_i = torch.cat([best_d, dist], 1), torch.cat([best_i, ids], 1)
            best_d, pos = cand_d.topk(k, dim=1, largest=False)
            best_i = cand_i.gather(1, pos)
        exact = (g[:, None, :] - train[best_i].float() / 255).norm(dim=-1)
        exact, order = exact.sort(dim=1)
        all_idx.append(best_i.gather(1, order).cpu())
        all_dist.append(exact.cpu())
    return torch.cat(all_idx).numpy(), torch.cat(all_dist).numpy().astype(np.float32)


def nn_figure(gen, train, idx, path, dist=None, title: str | None = None, scale: int = 2) -> Path:
    """Grilla de novedad: cada fila = muestra generada | vecino 1 | vecino 2 | ... (sin suavizado).

    gen: uint8 [M, H, W, 3]; train: uint8 [N, H, W, 3]; idx: [M, k] de nearest_neighbors.
    """
    from matplotlib.figure import Figure  # sin pyplot: no altera el backend del usuario

    gen, idx = np.asarray(gen), np.asarray(idx)
    m, k = idx.shape
    fig = Figure(figsize=((k + 1) * 1.1 * scale, m * 1.1 * scale))
    axes = fig.subplots(m, k + 1, squeeze=False)
    for r in range(m):
        row = [gen[r]] + [np.asarray(train[j]) for j in idx[r]]
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title("Generado" if c == 0 else f"Vecino {c}", fontsize=9)
            if c > 0 and dist is not None:
                ax.set_xlabel(f"d = {float(dist[r, c - 1]):.1f}", fontsize=7)
    fig.suptitle(title or "Novedad: cada muestra generada junto a sus vecinos más cercanos del "
                          "entrenamiento (L2 en píxeles)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # deja espacio al título
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    return path


# ---------------------------------------------------------------------------------------------
# 5. Generación y evaluación rápida durante el entrenamiento
# ---------------------------------------------------------------------------------------------

def to_u8(x: torch.Tensor) -> torch.Tensor:
    """Salida de G en [-1, 1] -> uint8 [0, 255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


@torch.no_grad()
def generate_u8(G: torch.nn.Module, y: torch.Tensor, lam: float = 1.0, device=None, seed: int = 0,
                batch: int = 128) -> torch.Tensor:
    """Genera un par por fila de y (multi-hot [N, 18]) con z ~ N(0, I) -> uint8 [N, 6, 96, 96] en CPU.

    Usa G en eval() (las estadísticas BN deben estar listas: standing_stats antes) y restaura el
    modo original al terminar.
    """
    device = torch.device(device or next(G.parameters()).device)
    was_training = G.training
    G.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i in range(0, len(y), batch):
        yb = torch.as_tensor(y[i:i + batch]).float().to(device)
        z = torch.randn(len(yb), G.z_dim, generator=g).to(device)
        out.append(to_u8(G(z, yb, lam)).cpu())
    G.train(was_training)
    return torch.cat(out)


def real_eval_features(path=REAL_FEATS_PATH, data_path=DATA_PATH, n: int = 2000, device=None,
                       seed: int = 0) -> dict:
    """Rasgos Inception de n pares reales FIJOS de entrenamiento (frente y espalda), en caché.

    Se calculan una sola vez y se guardan en `path` (data/cache/eval_real_feats.pt)."""
    path = Path(path)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    from .data import load_curated, to_tensor_pairs

    train, _ = load_curated(data_path)
    idx = np.sort(np.random.default_rng(seed).choice(len(train), size=min(n, len(train)), replace=False))
    x = to_tensor_pairs(train.subset(idx), "cpu")
    feats = {"front": inception_features(x[:, :3], device), "back": inception_features(x[:, 3:], device),
             "idx": idx}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    return feats


_CLASSIFIERS: dict[tuple[str, str], tuple] = {}


def _cached_classifier(path, device):
    key = (str(path), str(device))
    if key not in _CLASSIFIERS:
        from .type_classifier import load_classifier

        _CLASSIFIERS[key] = load_classifier(path, device)
    return _CLASSIFIERS[key]


def quick_eval(G: torch.nn.Module, lam: float, label_bank: torch.Tensor, device=None, n: int = 2000,
               seed: int = 123, real_feats_path=REAL_FEATS_PATH, clf_path=CLASSIFIER_PATH,
               data_path=DATA_PATH) -> dict:
    """Evaluación rápida para llamar cada pocos miles de iteraciones (G_ema con standing stats).

    Genera n pares con condiciones muestreadas de label_bank (las etiquetas de entrenamiento, así
    p_g(y) = p_d(y)), z ~ N(0, I) y el lam dado. Devuelve:
      kid_front, kid_back  KID contra 2000 reales fijos (rasgos en caché).
      type_map             mAP del clasificador de tipos (solo si existe runs/type_classifier/best.pt).
      coherence            HI de paleta medio de los pares generados.
    """
    device = torch.device(device or next(G.parameters()).device)
    real = real_eval_features(real_feats_path, data_path, device=device)
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, len(label_bank), (n,), generator=g)
    y = label_bank.cpu()[rows].float()
    x = generate_u8(G, y, lam, device, seed=seed)

    out = {}
    for view, sl in (("front", slice(0, 3)), ("back", slice(3, 6))):
        feats = inception_features(x[:, sl], device)
        out[f"kid_{view}"] = kid_from_feats(real[view], feats, subset_size=1000, subsets=50)[0]
    if Path(clf_path).exists():
        clf, _ = _cached_classifier(clf_path, device)
        from .type_classifier import predict_logits

        logits = predict_logits(clf, torch.cat([x[:, :3], x[:, 3:]]))
        out["type_map"] = type_metrics(logits, torch.cat([y, y]))["mAP"]
    xn = x.permute(0, 2, 3, 1).numpy()
    out["coherence"] = float(palette_hi_batch(xn[..., :3], xn[..., 3:]).mean())
    return out
