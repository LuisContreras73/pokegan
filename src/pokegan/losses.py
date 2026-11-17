"""Pérdidas y regularizadores.

Hinge (Lim & Ye 2017; SN-GAN; BigGAN):
    L_D = E_real[relu(1 − C(x))] + E_fake[relu(1 + C(G(z)))]   (+ ½·E_mis[relu(1 + C(x_mis))])
    L_G = −E[C(G(z))]
Negativos desajustados x_mis = (frente_i, espalda_j) de especies distintas: pares reales por vista pero
incoherentes como par ⇒ fuera del soporte de p_data ⇒ no sesgan el óptimo de G y obligan a D a juzgar
la coherencia frente/espalda.

Transición incondicional→condicional (Shahbazi et al. 2022, arXiv:2201.06578):
    λ_t = clip((t − T_s)/(T_e − T_s), 0, 1)

LeCam (Tseng et al. 2021, arXiv:2104.03310): con anclas α_R, α_F = EMA de E[C(real)], E[C(fake)],
    R_LC = E[relu(C(real) − α_F)²] + E[relu(α_R − C(fake))²]
evita que D separe reales y falsos de forma extrema (sobreajuste) con pocos datos.

Mode seeking (MSGAN, Mao et al. 2019, arXiv:1903.05628): L_ms = 1 / (mean|G(z1)−G(z2)| / mean|z1−z2| + ε)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def d_hinge(real: torch.Tensor, fake: torch.Tensor, mis: torch.Tensor | None = None,
            mis_weight: float = 0.5, mis_mask: torch.Tensor | None = None,
            fake_weight: float = 1.0, mis_pair_real: torch.Tensor | None = None) -> torch.Tensor:
    """Hinge de D con negativos desajustados opcionales.

    `mis_mask` (bool) descarta desajustados inválidos sin cambiar la forma del lote (sin sincronizar).

    Dos formas del término desajustado:
    - absoluta: w_m·relu(1 + C_mis). Un D constante siente dL/dc = w_f + w_m − w_r; con (1, 1, ½) vale +½
      y lo arrastra a C = −1 (colapso observado el 2026-11-16). Con GAN-CLS (1, ½, ½) se arregla al inicio,
      pero cuando D ya hunde los desajustados bajo −1 queda real 1 vs falso ½: deriva hacia +1 y, con hinge,
      el equilibrio de G se vuelve q ≤ 2p (puede abandonar modos).
    - por pares (`mis_pair_real` = logits de los reales cuyo frente usa cada desajustado):
      w_m·relu(1 − C_real,i + C_mis,i). Es relativista: el sesgo se cancela, así que no hay deriva en ningún
      régimen y con w_r = w_f = 1 el equilibrio sigue siendo q = p. Señal pura de coherencia.
    """
    loss = F.relu(1.0 - real).mean() + fake_weight * F.relu(1.0 + fake).mean()
    if mis is not None and mis.numel() > 0:
        h = F.relu(1.0 + mis) if mis_pair_real is None else F.relu(1.0 - mis_pair_real + mis)
        if mis_mask is None:
            loss = loss + mis_weight * h.mean()
        else:
            w = mis_mask.to(h.dtype)
            loss = loss + mis_weight * (h * w).sum() / w.sum().clamp_min(1.0)
    return loss


def g_hinge(fake: torch.Tensor) -> torch.Tensor:
    return -fake.mean()


def transition_lambda(step: int, t_start: int, t_end: int) -> float:
    if t_end <= t_start:
        return float(step >= t_start)
    return float(min(max((step - t_start) / (t_end - t_start), 0.0), 1.0))


class LeCam:
    """Anclas como tensores 0-d en el dispositivo (sin .item() por paso ⇒ sin sincronizar CPU-GPU).

    Debe aplicarse a los MISMOS logits que entrena la hinge: si solo se regulariza C_c = C_uc + proy.,
    D satisface LeCam volviendo la proyección anti-discriminativa (revisión adversarial, 2026-11-16).
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.ema_real: torch.Tensor | float | None = None
        self.ema_fake: torch.Tensor | float | None = None

    @torch.no_grad()
    def update(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        r, f = real.detach().mean(), fake.detach().mean()
        if self.ema_real is None:
            self.ema_real, self.ema_fake = r.clone(), f.clone()
        else:
            self.ema_real = self.decay * self.ema_real + (1 - self.decay) * r
            self.ema_fake = self.decay * self.ema_fake + (1 - self.decay) * f

    def penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        if self.ema_real is None:
            return real.new_zeros(())
        return F.relu(real - self.ema_fake).pow(2).mean() + F.relu(self.ema_real - fake).pow(2).mean()

    def state_dict(self) -> dict:
        f = lambda v: None if v is None else float(v)  # noqa: E731
        return {"decay": self.decay, "ema_real": f(self.ema_real), "ema_fake": f(self.ema_fake)}

    def load_state_dict(self, s: dict) -> None:
        self.decay, self.ema_real, self.ema_fake = s["decay"], s["ema_real"], s["ema_fake"]


def mode_seeking(x1: torch.Tensor, x2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor,
                 eps: float = 1e-5) -> torch.Tensor:
    lz = (x1 - x2).abs().mean() / (z1 - z2).abs().mean()
    return 1.0 / (lz + eps)


def derangement_other_species(species: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Índices j(i) ≠ i (ciclo sobre una permutación aleatoria) y máscara 'especie distinta'.

    Con ~1000 especies la colisión de especie es rara (~0.3 %); esas filas se descartan con la máscara.
    """
    n = species.shape[0]
    perm = torch.randperm(n, device=species.device)
    idx = torch.empty_like(perm)
    idx[perm] = perm.roll(-1)
    return idx, species[idx] != species


def alpha_shape_penalty(alpha: torch.Tensor, x: torch.Tensor | None = None, min_area: float = 240.0,
                        tv_target: float = 0.04) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularizadores de forma: área visible mínima (soporte) y perímetro (TV de α) acotado al nivel real.

    - Área visible (restricción de soporte): se mide sobre la imagen COMPUESTA x [B, 3V, H, W] ∈ [-1, 1]
      (no sobre α: los "vacíos" observados tienen α > 0 pero RGB casi blanco). Primer plano suave
      s = σ(50·((1 − min_c x) − 0.157)) (0.157 ⇔ umbral 235/255); penalización relu(min_area − Σs)/min_area.
      Ningún sprite real tiene < 240 px de primer plano.
    - Perímetro: TV(α) = mean|∇α|. Por la fórmula de co-área, ∫|∇α| = ∫₀¹ Per({α > t}) dt (perímetro medio
      de los conjuntos de nivel). Las máscaras reales tienen TV ≈ 0.04 y las generadas ≈ 0.12 (3× más
      perímetro: bordes rotos y fragmentos) ⇒ bisagra relu(TV − tv_target) hacia el nivel real.
    """
    tv = (alpha[..., 1:, :] - alpha[..., :-1, :]).abs().mean() + (alpha[..., :, 1:] - alpha[..., :, :-1]).abs().mean()
    p_tv = F.relu(tv - tv_target)
    if x is None:
        area = alpha.sum(dim=(-1, -2))
    else:
        b, c, h, w = x.shape
        dark = 1.0 - x.view(b, c // 3, 3, h, w).min(dim=2).values
        area = torch.sigmoid(50.0 * (dark - 0.157)).sum(dim=(-1, -2))  # pendiente alta: 9216 px blancos suman < 4
    p_area = F.relu(min_area - area).mean() / min_area
    return p_area, p_tv
