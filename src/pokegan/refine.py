"""Refinamiento en el espacio latente con NUESTRO discriminador: "el GAN es secretamente un modelo de energía".

Idea (Che et al. 2020, arXiv:2003.06060). Si D fuera óptimo, su logit sería d(x) = log p_data(x)/p_g(x), así que
    p_data(x) = p_g(x) · e^{d(x)}.
Como x = G(z) con z ~ N(0, I), la distribución corregida en el espacio latente es
    p*(z) ∝ N(z; 0, I) · e^{β d(G(z), y)}  =  e^{−E(z)},     E(z) = ½‖z‖² − β d(G(z), y),
y se muestrea con dinámica de Langevin:   z ← z − (ε/2) ∇E(z) + √ε · ξ,   ξ ~ N(0, I).
Con un D de proyección, d = C_uc + ⟨V y, φ⟩ estima log q(x, y)/p(x, y): el gradiente empuja a la vez hacia
"más real" y hacia "más del tipo y".

Variantes:
  - DOT (Tanaka 2019, arXiv:1910.06832): transporte óptimo con D como potencial de Kantorovich;
    z* = argmin_{z'} ‖z' − z‖ − d(G(z'))/K_eff, K_eff = constante de Lipschitz efectiva de d∘G.
  - SIR (muestreo-remuestreo por importancia, pariente de DRS, Azadi 2019): de K candidatos z_k se elige uno
    con probabilidad ∝ e^{β d_k}; si d fuera óptimo, cuando K → ∞ la muestra sigue p_g·e^{d} ∝ p_data.

Detalles prácticos:
  - D se entrenó sobre entradas aumentadas T(x) (DiffAugment) ⇒ su cociente de densidades vive en el espacio
    aumentado; promediamos d sobre varias transformaciones aleatorias (energía más suave).
  - D con hinge no da un log-cociente calibrado ⇒ temperatura β (hiperparámetro).
  - Todo usa solo G y D entrenados desde cero: nada preentrenado en la generación.
"""

from __future__ import annotations

import math

import torch

from .augment import diff_augment


def d_logit(D, x: torch.Tensor, y: torch.Tensor, n_aug: int = 4, w: float = 1.0) -> torch.Tensor:
    """Logit C_uc + w·proyección, promediado sobre n_aug transformaciones DiffAugment (0 = sin aumentar).

    w = 1 es el logit condicional de D. w > 1 es GUÍA DE TIPO (análoga a la guía por clasificador de difusión):
    la densidad inclinada es p_g(x|y)·r(x)^β·[p(y|x)/p_g(y|x)]^{βw}, porque la proyección estima el log-cociente
    del "clasificador" log p(y|x)/p_g(y|x). Compensa que G se entrenó con −C_uc − C_c = −2(C_uc + ½·proj).
    """
    if n_aug <= 0:
        uc, pj = D(x, y)
        return uc + w * pj
    xs = diff_augment(x.repeat(n_aug, 1, 1, 1))
    uc, pj = D(xs, y.repeat(n_aug, 1))
    return (uc + w * pj).view(n_aug, -1).mean(0)


def _prep(D):
    D.eval()
    D.requires_grad_(False)


def ddls(G, D, z: torch.Tensor, y: torch.Tensor, lam: float = 1.0, steps: int = 30, eps: float = 0.01,
         beta: float = 1.0, n_aug: int = 4, noise: float = 1.0) -> torch.Tensor:
    """Langevin en el espacio latente sobre E(z) = ½‖z‖² − β·d(G(z), y). Devuelve z refinado."""
    _prep(D)
    z = z.detach().clone()
    for _ in range(steps):
        z.requires_grad_(True)
        e = 0.5 * z.pow(2).sum(1) - beta * d_logit(D, G(z, y, lam), y, n_aug)
        (g,) = torch.autograd.grad(e.sum(), z)
        with torch.no_grad():
            z = z - 0.5 * eps * g + noise * math.sqrt(eps) * torch.randn_like(z)
    return z.detach()


@torch.no_grad()
def estimate_k_eff(G, D, y_bank: torch.Tensor, lam: float = 1.0, n: int = 256, n_aug: int = 4) -> float:
    """K_eff ≈ máx |d(G(z1)) − d(G(z2))| / ‖z1 − z2‖ sobre pares aleatorios con la misma condición (DOT)."""
    _prep(D)
    dev = next(G.parameters()).device
    y = y_bank[torch.randint(len(y_bank), (n,))].to(dev).float()
    z1, z2 = torch.randn(n, G.z_dim, device=dev), torch.randn(n, G.z_dim, device=dev)
    d1, d2 = d_logit(D, G(z1, y, lam), y, n_aug), d_logit(D, G(z2, y, lam), y, n_aug)
    return float(((d1 - d2).abs() / (z1 - z2).norm(dim=1)).max())


def dot(G, D, z: torch.Tensor, y: torch.Tensor, k_eff: float, lam: float = 1.0, steps: int = 30,
        lr: float = 0.01, n_aug: int = 4) -> torch.Tensor:
    """DOT latente: minimiza ‖z' − z‖ − d(G(z'), y)/K_eff con Adam desde z' = z."""
    _prep(D)
    z0 = z.detach()
    zp = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([zp], lr=lr, betas=(0.0, 0.9))
    for _ in range(steps):
        loss = (zp - z0).norm(dim=1) - d_logit(D, G(zp, y, lam), y, n_aug) / k_eff
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    return zp.detach()


def chunk_mask(z_dim: int, chunks: tuple[int, ...] | None, n_chunks: int = 6, device=None) -> torch.Tensor:
    """Máscara [z_dim] que deja mover solo algunos trozos del z jerárquico (0 = silueta 3×3 … 5 = cabezas 96 px)."""
    m = torch.ones(z_dim, device=device)
    if chunks is not None:
        m.zero_()
        size = z_dim // n_chunks
        for c in chunks:
            m[c * size:(c + 1) * size] = 1
    return m


def dgflow(G, D, z: torch.Tensor, y: torch.Tensor, lam: float = 1.0, steps: int = 25, eta: float = 0.1,
           gamma: float = 0.01, beta: float = 1.0, n_aug: int = 4, chunks: tuple[int, ...] | None = None,
           w: float = 1.0) -> torch.Tensor:
    """DGflow-KL en el latente (Ansari et al. 2021, arXiv:2012.00780): z ← z + η∇_z[β d(G(z),y)] + √(2ηγ)·ξ.

    Es el flujo de gradiente de la KL(p_g‖p_data) estimada por D, pulled back al espacio latente: versión corta
    y local de DDLS (sin el término del prior). `chunks` limita el movimiento a trozos de z (p. ej. (0,1,2) =
    solo la estructura gruesa: corrige la silueta conservando paleta y detalle).
    """
    _prep(D)
    mask = chunk_mask(z.shape[1], chunks, device=z.device)
    z = z.detach().clone()
    for _ in range(steps):
        z.requires_grad_(True)
        s = beta * d_logit(D, G(z, y, lam), y, n_aug, w)
        (g,) = torch.autograd.grad(s.sum(), z)
        with torch.no_grad():
            z = z + mask * (eta * g + math.sqrt(2 * eta * gamma) * torch.randn_like(z))
    return z.detach()


def truncate(z: torch.Tensor, threshold: float, chunks: tuple[int, ...] | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Truco de truncamiento (BigGAN): re-muestrea |z_i| > umbral; con `chunks`, solo en esos trozos de z."""
    z = z.clone()
    mask = chunk_mask(z.shape[1], chunks, device=z.device).bool()
    bad = (z.abs() > threshold) & mask
    while bad.any():
        z[bad] = torch.randn(int(bad.sum()), generator=generator).to(z.device) if generator else torch.randn(int(bad.sum()), device=z.device)
        bad = (z.abs() > threshold) & mask
    return z


@torch.no_grad()
def platt_beta(D, x_real: torch.Tensor, y_real: torch.Tensor, x_fake: torch.Tensor, y_fake: torch.Tensor,
               n_aug: int = 8) -> tuple[float, float]:
    """Calibra el logit de D (Platt): P(real | s) = σ(a·s + b) por regresión logística.

    Reales = especies NO vistas (validación; D está sobreajustado a las de entrenamiento). Devuelve (a, b):
    a es la temperatura β a usar en DDLS/SIR (LeCam comprime los logits, así que se espera a > 1).
    """
    from sklearn.linear_model import LogisticRegression

    _prep(D)
    s = []
    for x, y in ((x_real, y_real), (x_fake, y_fake)):
        s.append(torch.cat([d_logit(D, x[i:i + 64], y[i:i + 64], n_aug) for i in range(0, len(x), 64)]).cpu())
    X = torch.cat(s).numpy()[:, None]
    t = [1] * len(x_real) + [0] * len(x_fake)
    lr = LogisticRegression(C=10.0).fit(X, t)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def count_fragments(x_u8: torch.Tensor, white_thr: int = 235, min_area: int = 4) -> torch.Tensor:
    """Componentes 8-conexas de primer plano (≥ min_area px) por imagen uint8 [N, 3, H, W]: 1 = cuerpo único."""
    import numpy as np
    from scipy import ndimage

    fg = (x_u8.min(1).values < white_thr).numpy()
    out = []
    for f in fg:
        lab, n = ndimage.label(f, structure=np.ones((3, 3)))
        areas = np.bincount(lab.ravel())[1:]
        out.append(int((areas >= min_area).sum()))
    return torch.tensor(out)


@torch.no_grad()
def sir(G, D, y: torch.Tensor, k: int = 8, lam: float = 1.0, beta: float = 1.0, n_aug: int = 4,
        generator: torch.Generator | None = None, w: float = 1.0,
        trunc_coarse: float | None = None) -> torch.Tensor:
    """Para cada condición: K candidatos z_k y se elige uno con prob. ∝ e^{β d_k}. Devuelve z elegidos [N, z_dim]."""
    _prep(D)
    dev = y.device
    n = len(y)
    zs = torch.randn(k, n, G.z_dim, generator=generator).to(dev)
    if trunc_coarse is not None:  # candidatos con prior más frío en z0–z2 (silueta)
        zs = truncate(zs.view(k * n, -1), trunc_coarse, (0, 1, 2), generator).view(k, n, -1)
    d = torch.stack([d_logit(D, G(zs[i], y, lam), y, n_aug, w) for i in range(k)])  # k, n
    pick = torch.multinomial(torch.softmax(beta * d, 0).T, 1).squeeze(1)       # n
    return zs[pick, torch.arange(n, device=dev)]


@torch.no_grad()
def support_resample(G, z: torch.Tensor, y: torch.Tensor, lam: float = 1.0, min_px: int = 240,
                     max_tries: int = 20, white_thr: int = 235,
                     generator: torch.Generator | None = None, trunc_coarse: float | None = None) -> torch.Tensor:
    """Restricción de soporte: re-muestrea z (misma condición) si alguna vista tiene < min_px px de primer plano.

    Ningún sprite real tiene menos de 240 px de primer plano ⇒ esas muestras están FUERA del soporte de p_data.
    Restringir p_g a A ⊇ supp(p_data) y renormalizar, p' = p_g·1_A / p_g(A), cumple
        KL(p_data ‖ p') = KL(p_data ‖ p_g) + log p_g(A) ≤ KL(p_data ‖ p_g),
    es decir, el rechazo reduce la divergencia exactamente en −log p_g(A). Filtro determinista: no usa ninguna red.
    """
    z = z.clone()
    for _ in range(max_tries):
        x = G(z, y, lam)
        u8 = ((x.clamp(-1, 1) + 1) * 127.5).round()
        fg = (u8.view(len(z), 2, 3, *u8.shape[-2:]).min(2).values < white_thr).sum((-1, -2)).min(1).values
        bad = fg < min_px
        if not bad.any():
            break
        new = torch.randn(int(bad.sum()), z.shape[1], generator=generator).to(z.device)
        if trunc_coarse is not None:
            new = truncate(new, trunc_coarse, (0, 1, 2), generator)
        z[bad] = new
    return z
