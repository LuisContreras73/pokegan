"""Bloques de construcción (estilo BigGAN / SN-GAN).

- Normalización espectral (SN) en todas las capas lineales/convolucionales: acota la constante de
  Lipschitz de cada capa por 1 dividiendo W entre su mayor valor singular σ(W), estimado con una
  iteración de potencia por paso.
- BatchNorm condicional (CBN): la ganancia y el sesgo de BN se predicen a partir de [z_ℓ; e(y)].
- Auto-atención (SAGAN) para dependencias de largo alcance (p. ej. simetrías del cuerpo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

SN_EPS = 1e-6


def _ortho(m: nn.Module) -> nn.Module:
    nn.init.orthogonal_(m.weight)
    if getattr(m, "bias", None) is not None:
        nn.init.zeros_(m.bias)
    return m


def snconv(cin: int, cout: int, k: int = 3, padding: int | None = None, bias: bool = True) -> nn.Module:
    p = k // 2 if padding is None else padding
    return spectral_norm(_ortho(nn.Conv2d(cin, cout, k, padding=p, bias=bias)), eps=SN_EPS)


def snlinear(fin: int, fout: int, bias: bool = True) -> nn.Module:
    return spectral_norm(_ortho(nn.Linear(fin, fout, bias=bias)), eps=SN_EPS)


class ConditionalBN(nn.Module):
    """BN sin afinidad propia + γ = 1 + W_γ c, β = W_β c (BigGAN)."""

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=1e-5)
        self.gain = snlinear(cond_dim, num_features, bias=False)
        self.bias = snlinear(cond_dim, num_features, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g = 1.0 + self.gain(c)[:, :, None, None]
        b = self.bias(c)[:, :, None, None]
        return self.bn(x) * g + b


class SelfAttention(nn.Module):
    """Auto-atención de SAGAN/BigGAN: x + γ·o, con γ inicializado en 0 (empieza como identidad)."""

    def __init__(self, ch: int):
        super().__init__()
        self.theta = snconv(ch, ch // 8, 1, bias=False)
        self.phi = snconv(ch, ch // 8, 1, bias=False)
        self.g = snconv(ch, ch // 2, 1, bias=False)
        self.o = snconv(ch // 2, ch, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        theta = self.theta(x).flatten(2)                       # B, C/8, HW
        phi = F.max_pool2d(self.phi(x), 2).flatten(2)          # B, C/8, HW/4
        g = F.max_pool2d(self.g(x), 2).flatten(2)              # B, C/2, HW/4
        attn = torch.softmax(theta.transpose(1, 2) @ phi, -1)  # B, HW, HW/4
        o = (g @ attn.transpose(1, 2)).view(b, c // 2, h, w)
        return x + self.gamma * self.o(o)


class GBlock(nn.Module):
    """Bloque residual de G: CBN-ReLU-Up(×2 vecino)-Conv3-CBN-ReLU-Conv3 + atajo Up-Conv1."""

    def __init__(self, cin: int, cout: int, cond_dim: int):
        super().__init__()
        self.bn1 = ConditionalBN(cin, cond_dim)
        self.conv1 = snconv(cin, cout, 3)
        self.bn2 = ConditionalBN(cout, cond_dim)
        self.conv2 = snconv(cout, cout, 3)
        self.sc = snconv(cin, cout, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(F.relu(self.bn1(x, c)), scale_factor=2, mode="nearest")
        h = self.conv2(F.relu(self.bn2(self.conv1(h), c)))
        return h + self.sc(F.interpolate(x, scale_factor=2, mode="nearest"))


class DBlock(nn.Module):
    """Bloque residual de D (pre-activación). El primero ('optimizado') no aplica ReLU a la entrada."""

    def __init__(self, cin: int, cout: int, down: bool = True, preact: bool = True):
        super().__init__()
        self.conv1 = snconv(cin, cout, 3)
        self.conv2 = snconv(cout, cout, 3)
        self.sc = snconv(cin, cout, 1) if (cin != cout or down) else None
        self.down, self.preact = down, preact

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x) if self.preact else x
        h = self.conv2(F.relu(self.conv1(h)))
        if self.down:
            h = F.avg_pool2d(h, 2)
        s = x
        if self.preact:  # atajo: conv -> pool
            s = self.sc(s) if self.sc is not None else s
            s = F.avg_pool2d(s, 2) if self.down else s
        else:            # bloque de entrada: pool -> conv
            s = F.avg_pool2d(s, 2) if self.down else s
            s = self.sc(s) if self.sc is not None else s
        return h + s
