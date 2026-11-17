"""Evaluador de tipos con *linear probe* sobre un backbone PREENTRENADO y CONGELADO (solo evaluación).

Motivo: el clasificador desde cero (type_classifier.py) sobreajusta con ~800 criaturas distintas y en
especies no vistas apenas llega a mAP ≈ 0.24 (azar ≈ 0.09): un evaluador débil vuelve poco concluyente
la métrica de condicionamiento. Las features genéricas de un modelo preentrenado (DINOv3 / ConvNeXt)
+ una capa lineal entrenada con sprites reales generalizan mucho mejor a especies nuevas.

El probe NUNCA participa en el entrenamiento del GAN (sería optimizar la métrica);
solo mide. Se declara junto con Inception (KID/FID).

Interfaz compatible con type_classifier: forward(x en [-1,1], 96 px) -> logits [B, 18]; se guarda con
'kind': 'probe' y load_classifier() lo reconoce.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import NUM_TYPES

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
BACKBONES = {
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",  # transformers
    "convnext": "convnext_small.fb_in22k_ft_in1k",         # timm
}


class Backbone(nn.Module):
    """Extractor congelado: [-1,1] 96 px → reescala a 224 (bicúbica) → normaliza ImageNet → vector."""

    def __init__(self, name: str = "dinov3"):
        super().__init__()
        self.name = name
        if name == "dinov3":
            from transformers import AutoModel

            self.net = AutoModel.from_pretrained(BACKBONES[name])
            self.dim = self.net.config.hidden_size
        elif name == "convnext":
            import timm

            self.net = timm.create_model(BACKBONES[name], pretrained=True, num_classes=0)
            self.dim = self.net.num_features
        else:
            raise ValueError(name)
        self.net.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() + 1) / 2
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self.mean) / self.std
        if self.name == "dinov3":
            return self.net(pixel_values=x).pooler_output.float()
        return self.net(x).float()


class TypeProbe(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, NUM_TYPES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@torch.no_grad()
def extract(backbone: nn.Module, x_u8: torch.Tensor, device, batch: int = 128, flip: bool = False) -> torch.Tensor:
    out = []
    for i in range(0, len(x_u8), batch):
        x = x_u8[i:i + batch].to(device).float() / 127.5 - 1
        if flip:
            x = x.flip(-1)
        out.append(backbone(x).cpu())
    return torch.cat(out)


def fit_head(f_tr: torch.Tensor, y_tr: torch.Tensor, f_va: torch.Tensor, y_va: torch.Tensor,
             device, epochs: int = 3000, lr: float = 1e-3, wd: float = 1e-4) -> tuple[nn.Linear, dict]:
    """Regresión logística multi-etiqueta (BCE con pos_weight) en lote completo; elige la mejor época por mAP val."""
    from .evaluate import type_metrics

    mu, sd = f_tr.mean(0, keepdim=True), f_tr.std(0, keepdim=True).clamp_min(1e-6)
    ftr, fva = ((f_tr - mu) / sd).to(device), ((f_va - mu) / sd).to(device)
    ytr = y_tr.to(device)
    n_t = ytr.sum(0)
    pos_w = ((len(ytr) - n_t) / n_t.clamp_min(1)).clamp(1, 10)
    head = nn.Linear(ftr.shape[1], NUM_TYPES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        loss = F.binary_cross_entropy_with_logits(head(ftr), ytr, pos_weight=pos_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 49:
            with torch.no_grad():
                m = type_metrics(head(fva).cpu(), y_va)["mAP"]
            hist.append((ep + 1, float(loss), m))
            if m > best:
                best, best_state = m, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    # plegar la estandarización en la capa lineal: W' = W/sd, b' = b − W'·mu
    with torch.no_grad():
        W = head.weight / sd.to(device)
        b = head.bias - (W * mu.to(device)).sum(1)
        head.weight.copy_(W)
        head.bias.copy_(b)
    return head.cpu(), {"val_mAP": best, "history": hist}


def save_probe(path: str | Path, backbone_name: str, head: nn.Linear, thresholds: np.ndarray, metrics: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "probe", "backbone": backbone_name, "head": head.state_dict(),
                "thresholds": np.asarray(thresholds, np.float32), "val_metrics": metrics}, path)


def load_probe(ckpt: dict, device) -> tuple[TypeProbe, np.ndarray]:
    bb = Backbone(ckpt["backbone"])
    model = TypeProbe(bb, bb.dim)
    model.head.load_state_dict(ckpt["head"])
    return model.to(device).eval(), np.asarray(ckpt["thresholds"], dtype=np.float32)
