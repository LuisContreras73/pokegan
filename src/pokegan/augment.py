"""DiffAugment PAREADO (Zhao et al. 2020, arXiv:2006.10738), adaptado a pares frente/espalda.

Idea: D sobreajusta con pocos datos. Si aplicamos la MISMA familia de transformaciones T diferenciables a
reales y falsos (en los pasos de D y de G), D ve T(x_real) vs T(G(z)) y G recibe gradiente a través de T.
Como T se aplica a ambos, el óptimo sigue siendo p_g = p_data siempre que T sea invertible en distribución
(traslación y cutout lo son en la práctica: no "filtran" al generador).

Adaptación clave: se opera sobre el tensor de 6 canales [frente; espalda] ⇒ cada par recibe los MISMOS
parámetros aleatorios en ambas vistas. Si se transformaran por separado, D vería pares "reales"
desalineados y aprendería que la incoherencia es aceptable.

Sin aumentos de color: el color es la evidencia principal del tipo (Fuego≈rojo, Agua≈azul).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rand_translation(x: torch.Tensor, ratio: float = 0.125, fill: float = 1.0) -> torch.Tensor:
    """Desplaza cada muestra hasta ±ratio·tamaño píxeles; lo que entra por el borde es blanco (+1)."""
    b, _, h, w = x.shape
    sx, sy = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, (b, 1, 1), device=x.device)
    ty = torch.randint(-sy, sy + 1, (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=x.device), torch.arange(h, device=x.device),
        torch.arange(w, device=x.device), indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    xp = F.pad(x, [1, 1, 1, 1], value=fill)
    return xp.permute(0, 2, 3, 1)[gb, gx, gy].permute(0, 3, 1, 2)


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Anula (valor 0 = gris en [-1,1], como el código oficial) un cuadrado aleatorio de ratio·tamaño."""
    b, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=x.device), torch.arange(ch, device=x.device),
        torch.arange(cw, device=x.device), indexing="ij",
    )
    gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


def diff_augment(x: torch.Tensor, translation: float = 0.125, cutout: float = 0.5) -> torch.Tensor:
    """Política 'translation,cutout' (la que DiffAugment usa con BigGAN condicional)."""
    if translation > 0:
        x = rand_translation(x, translation)
    if cutout > 0:
        x = rand_cutout(x, cutout)
    return x


def flip_pairs(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Volteo horizontal CONJUNTO de ambas vistas, solo para datos reales (aumento a nivel dataset).

    No se usa dentro de la tubería de D: un volteo con prob. p allí filtra la orientación (w = 1−2p).
    """
    if p <= 0:
        return x
    m = torch.rand(x.shape[0], device=x.device) < p
    return torch.where(m[:, None, None, None], x.flip(-1), x)
