"""Promedio exponencial (EMA) de los pesos de G y estadísticas BN 'standing' (BigGAN, App. C.1).

EMA: θ_ema ← d·θ_ema + (1−d)·θ. Promedia la trayectoria oscilante del juego minimax y da muestras
más limpias que el G "vivo". Los buffers (estadísticas BN, vectores u/v de la norma espectral) se COPIAN.

Standing stats: las estadísticas BN del EMA no corresponden a sus pesos promediados ⇒ antes de muestrear
se recalculan con varias pasadas hacia adelante en modo train (momentum=None = promedio acumulado).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, start: int = 1000):
        self.decay, self.start = decay, start
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = self.decay if step >= self.start else 0.0  # antes de 'start': copia exacta
        live = dict(model.named_parameters())
        for k, p in self.module.named_parameters():
            p.lerp_(live[k], 1.0 - d)
        live_b = dict(model.named_buffers())
        for k, b in self.module.named_buffers():
            b.copy_(live_b[k])


@torch.no_grad()
def standing_stats(G: nn.Module, ys: torch.Tensor, n_batches: int = 32, batch: int = 64,
                   lam: float = 1.0) -> None:
    """Recalcula las estadísticas BN de G con z ~ N(0, I) y condiciones tomadas de `ys`; deja G en eval()."""
    device = next(G.parameters()).device
    ys = ys.to(device).float()
    bns = [m for m in G.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    G.train()
    for _ in range(n_batches):
        idx = torch.randint(len(ys), (batch,), device=device)
        G(torch.randn(batch, G.z_dim, device=device), ys[idx], lam)
    for m, mo in zip(bns, momenta):
        m.momentum = mo
    G.eval()
