"""Discriminador asistido por visión (Vision-aided GAN, Kumari et al. 2022, arXiv:2112.09130).

Backbone PREENTRENADO y CONGELADO (ConvNeXt-T ImageNet-22k, de timm) + cabeza pequeña entrenable con proyección
de tipo. Se SUMA al D propio (entrenado desde cero): G sigue entrenándose desde cero, pero recibe gradiente de
rasgos genéricos que sí codifican forma/estructura (nuestro D mide sobre todo nitidez; ver results/latente).

⚠️ Reglas del curso: si "desde cero" se interpreta de forma estricta, este componente NO cumple (usa pesos
preentrenados dentro de un discriminador). Por eso el modelo con este D se presenta como VARIANTE declarada,
al lado del modelo puro desde cero.

Entrada: par [B, 6, 96, 96] en [-1, 1] → cada vista por separado al backbone → [f_frente; f_espalda] → cabeza.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import snlinear


class VisionD(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny.fb_in22k_ft_in1k", res: int = 160, n_types: int = 18,
                 hidden: int = 512):
        super().__init__()
        import timm

        self.net = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.net.eval().requires_grad_(False)
        cfg = self.net.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.res = res
        dim = self.net.num_features
        self.head = nn.Sequential(snlinear(2 * dim, hidden), nn.LeakyReLU(0.2), snlinear(hidden, hidden),
                                  nn.LeakyReLU(0.2))
        self.psi = snlinear(hidden, 1)
        self.proj = snlinear(n_types, hidden, bias=False)

    def train(self, mode: bool = True):  # el backbone queda SIEMPRE en eval (congelado)
        super().train(mode)
        self.net.eval()
        return self

    def features(self, x6: torch.Tensor) -> torch.Tensor:
        b = x6.shape[0]
        v = torch.cat([x6[:, :3], x6[:, 3:]], 0)
        v = F.interpolate((v + 1) / 2, size=self.res, mode="bilinear", align_corners=False)
        f = self.net((v - self.mean) / self.std)
        return torch.cat([f[:b], f[b:]], 1)

    def forward(self, x6: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.head(self.features(x6))
        return self.psi(h).squeeze(1) + (self.proj(y) * h).sum(1)

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("net.")]
