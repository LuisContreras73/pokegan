"""Discriminador de pares con proyección multi-hot.

   x = [frente; espalda] (6 canales, 96×96)  — ver ambas vistas a la vez permite juzgar coherencia
   → DBlock 32@48 → DBlock 64@24 → SelfAttention → 128@12 → 256@6 → 512@3 → 512 → ReLU → Σ espacial = φ(x)
   salida incondicional  ψ(φ) = w·φ + b
   proyección             ⟨V y, φ⟩ = Σ_{t ∈ tipos(y)} ⟨v_t, φ⟩   (multi-hot CRUDO = suma por tipo)

   C_uc(x) = ψ(φ(x)) ;  C_c(x, y) = ψ(φ(x)) + ⟨V y, φ(x)⟩

La suma es la forma exacta del log-cociente óptimo log q(y|x)/p(y|x) cuando q(c|x) es log-lineal sobre
conjuntos de tipos (extiende Miyato & Koyama 2018). SN en todas las capas; sin BatchNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DBlock, SelfAttention, snlinear


class Discriminator(nn.Module):
    def __init__(self, ch: int = 32, n_types: int = 18, attn: bool = True, in_ch: int = 6):
        super().__init__()
        w = [ch, 2 * ch, 4 * ch, 8 * ch, 16 * ch, 16 * ch]
        self.b0 = DBlock(in_ch, w[0], down=True, preact=False)  # 48
        self.b1 = DBlock(w[0], w[1])                              # 24
        self.attn = SelfAttention(w[1]) if attn else None
        self.rest = nn.Sequential(
            DBlock(w[1], w[2]),              # 12
            DBlock(w[2], w[3]),              # 6
            DBlock(w[3], w[4]),              # 3
            DBlock(w[4], w[5], down=False),  # 3
        )
        self.psi = snlinear(w[5], 1)
        self.proj = snlinear(n_types, w[5], bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.b1(self.b0(x))
        if self.attn is not None:
            h = self.attn(h)
        return F.relu(self.rest(h)).sum(dim=(2, 3))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve (C_uc, término de proyección), cada uno [B]. C_c = C_uc + proyección."""
        phi = self.features(x)
        return self.psi(phi).squeeze(1), (self.proj(y) * phi).sum(1)
