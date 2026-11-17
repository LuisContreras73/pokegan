"""Generador BigGAN-lite-96 de dos vistas.

   z ∈ R^120 ──split en 6 trozos de 20──┐           y ∈ {0,1}^18 (1 o 2 tipos)
                                        │           e = Eᵀy  (suma de embeddings por tipo)
                                        │           ê = λ_t · RMSNorm(e)
   z0 → SNLinear → 512×3×3              │
        → GBlock(z1,ê) 512@6 → GBlock(z2,ê) 256@12 → GBlock(z3,ê) 128@24 → SelfAttention
        → GBlock(z4,ê) 64@48 ─┬→ cabeza FRENTE (GBlock(z5,ê) 32@96 → BN-ReLU-Conv → RGB+α)
                              └→ cabeza ESPALDA (misma estructura, pesos propios)
   Composición alfa por vista: x_v = σ(a_v)·tanh(rgb_v) + (1−σ(a_v))·1  → fondo blanco exacto.
   Salida: [B, 6, 96, 96] = [frente RGB; espalda RGB] en [-1, 1].

Un único z alimenta TODO (tronco y ambas cabezas) ⇒ ambas vistas son la misma criatura.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GBlock, SelfAttention, snconv, snlinear


class ViewHead(nn.Module):
    """Cabeza por vista: 48→96 px y salida RGB + máscara alfa compuesta sobre blanco."""

    def __init__(self, cin: int, cmid: int, cond_dim: int):
        super().__init__()
        self.block = GBlock(cin, cmid, cond_dim)
        self.bn = nn.BatchNorm2d(cmid)
        self.out = snconv(cmid, 4, 3)

    def forward(self, h: torch.Tensor, c: torch.Tensor, return_alpha: bool = False):
        o = self.out(F.relu(self.bn(self.block(h, c))))
        rgb, alpha = torch.tanh(o[:, :3]), torch.sigmoid(o[:, 3:4])
        x = alpha * rgb + (1.0 - alpha)  # blanco = +1
        return (x, alpha) if return_alpha else x


class Generator(nn.Module):
    def __init__(self, ch: int = 32, z_dim: int = 120, n_chunks: int = 6, emb_dim: int = 128,
                 n_types: int = 18, attn: bool = True, spatial_z: bool = False):
        super().__init__()
        assert z_dim % n_chunks == 0 and n_chunks == 6, "z jerárquico: stem + 4 bloques + cabezas"
        self.z_dim, self.chunk = z_dim, z_dim // n_chunks
        w = [16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch]  # 3, 6, 12, 24, 48 px
        cond_dim = self.chunk + emb_dim
        self.embed = nn.Linear(n_types, emb_dim, bias=False)  # e = Eᵀy ⇒ suma de embeddings
        nn.init.normal_(self.embed.weight, std=1.0)
        self.stem = snlinear(self.chunk, w[0] * 3 * 3)
        self.blocks = nn.ModuleList([GBlock(w[i], w[i + 1], cond_dim) for i in range(4)])
        self.attn = SelfAttention(w[3]) if attn else None  # 24×24
        self.heads = nn.ModuleList([ViewHead(w[4], ch, cond_dim) for _ in range(2)])
        # Ruta espacial para z1..z4 (corrige la subutilización medida: por la BN condicional solo modulan canales).
        # z_l → mapa H×W sumado a la salida de su bloque con escala por canal inicializada en 0 (arranca idéntico).
        self.spatial_z = spatial_z
        if spatial_z:
            self.zmap = nn.ModuleList([nn.Linear(self.chunk, (6 * 2 ** i) ** 2) for i in range(4)])
            self.zmap_scale = nn.ParameterList([nn.Parameter(torch.zeros(w[i + 1])) for i in range(4)])

    def type_embedding(self, y: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        e = self.embed(y)
        return lam * e * torch.rsqrt(e.pow(2).mean(1, keepdim=True) + 1e-8)

    def forward(self, z: torch.Tensor, y: torch.Tensor, lam: float = 1.0, return_alpha: bool = False):
        """[B, 6, 96, 96]; con return_alpha=True también devuelve las máscaras α [B, 2, 96, 96] (frente, espalda)."""
        zs = torch.split(z, self.chunk, dim=1)
        e = self.type_embedding(y, lam)
        h = self.stem(zs[0]).view(z.shape[0], -1, 3, 3)
        for i, block in enumerate(self.blocks):
            h = block(h, torch.cat([zs[i + 1], e], 1))
            if self.spatial_z:
                r = h.shape[-1]
                h = h + self.zmap_scale[i].view(1, -1, 1, 1) * self.zmap[i](zs[i + 1]).view(-1, 1, r, r)
            if i == 2 and self.attn is not None:
                h = self.attn(h)
        c = torch.cat([zs[5], e], 1)
        if return_alpha:
            outs = [head(h, c, True) for head in self.heads]
            return torch.cat([o[0] for o in outs], 1), torch.cat([o[1] for o in outs], 1)
        return torch.cat([head(h, c) for head in self.heads], 1)
