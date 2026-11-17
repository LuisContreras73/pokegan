"""Discriminadores auxiliares (entrenados DESDE CERO) que miran la FORMA, no solo la nitidez.

Diagnóstico (docs/matematica.md §10c): el D de 96 px juzga sobre todo nitidez y es ciego a la forma
(añadir fragmentos cuesta 3 % de la brecha; desenfocar, 94 %). Dos remedios clásicos:

1. Multiescala (pix2pixHD, Wang et al. 2018; MSG-GAN, Karnewar & Wang 2020): el mismo tipo de D sobre el par
   reducido a 48 y 24 px. A baja resolución un sprite ES su silueta: el D pequeño obliga a acertar la forma global.
2. Siluetas: un D sobre la máscara de primer plano (frente, espalda). Juzga topología: extremidades, fragmentos,
   huecos. Para los falsos la máscara es diferenciable: s = σ(50·((1 − min_c x) − 0.157)).

Todos usan normalización espectral, salida ψ(φ) + ⟨V y, φ⟩ (proyección multi-hot) y suma global.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DBlock, snlinear


class ScaleD(nn.Module):
    """D residual genérico para entradas de res px (res ∈ {24, 48, 96}) con in_ch canales."""

    def __init__(self, ch: int = 32, res: int = 48, in_ch: int = 6, n_types: int = 18):
        super().__init__()
        n_down = {24: 3, 48: 4, 96: 5}[res]          # hasta 3×3
        widths = [min(ch * 2 ** i, 8 * ch) for i in range(n_down + 1)]
        blocks = [DBlock(in_ch, widths[0], down=True, preact=False)]
        for i in range(1, n_down):
            blocks.append(DBlock(widths[i - 1], widths[i]))
        blocks.append(DBlock(widths[n_down - 1], widths[n_down], down=False))
        self.blocks = nn.Sequential(*blocks)
        self.psi = snlinear(widths[n_down], 1)
        self.proj = snlinear(n_types, widths[n_down], bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phi = F.relu(self.blocks(x)).sum(dim=(2, 3))
        return self.psi(phi).squeeze(1), (self.proj(y) * phi).sum(1)


def soft_fg_mask(x6: torch.Tensor) -> torch.Tensor:
    """Máscara de primer plano diferenciable [B, 2, H, W] desde el par [B, 6, H, W] en [-1, 1].

    Umbral 0.157 ⇔ min(RGB) < 235 (el mismo de las métricas); pendiente 50 ⇒ casi binaria pero con gradiente.
    """
    b, c, h, w = x6.shape
    dark = 1.0 - x6.view(b, c // 3, 3, h, w).min(dim=2).values
    return torch.sigmoid(50.0 * (dark - 0.157))


class MultiScaleD(nn.Module):
    """Conjunto de discriminadores auxiliares: par a 48 y 24 px + siluetas a 48 px."""

    def __init__(self, ch: int = 32, scales: tuple[int, ...] = (48, 24), mask: bool = True, n_types: int = 18):
        super().__init__()
        self.scales = tuple(scales)
        self.ds = nn.ModuleDict({str(r): ScaleD(ch, r, 6, n_types) for r in self.scales})
        self.mask_d = ScaleD(ch, 48, 2, n_types) if mask else None

    def forward(self, x6: torch.Tensor, y: torch.Tensor, x6_mask: torch.Tensor | None = None) -> list:
        """Lista de (C_uc, proy) por discriminador. x6_mask: par para la silueta (sin cutout), por defecto x6."""
        out = []
        for r in self.scales:
            xr = F.avg_pool2d(x6, x6.shape[-1] // r)
            out.append(self.ds[str(r)](xr, y))
        if self.mask_d is not None:
            m = soft_fg_mask(x6 if x6_mask is None else x6_mask)
            out.append(self.mask_d(F.avg_pool2d(m, 2) * 2 - 1, y))
        return out
