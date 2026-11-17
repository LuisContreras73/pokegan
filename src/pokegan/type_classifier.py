"""Clasificador de tipos DESDE CERO, usado solo para EVALUAR el GAN (protocolo "GAN-test").

Se entrena con sprites reales de las especies de entrenamiento (frentes y espaldas como muestras
separadas, 96 px) y se valida en las especies retenidas. Nunca participa en el entrenamiento del
GAN: usarlo como pérdida sería circular (el generador optimizaría la métrica).

Decisiones:
  - ResNet-18 de timm con pretrained=False (pesos aleatorios, nada de ImageNet).
  - Multi-etiqueta: BCEWithLogits con pos_weight_t = clip((N - n_t) / n_t, 1, 10) para que los
    tipos raros (p. ej. Hielo, Hada) no se ignoren.
  - Aumentos SOLO geométricos: traslación ±8 px con relleno blanco y volteo horizontal. Nada de
    color: el color es justamente la evidencia del tipo.
  - Parada temprana por mAP macro en validación y umbrales por clase que maximizan F1 en validación.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import PairData
from .types import NUM_TYPES


def make_classifier() -> torch.nn.Module:
    """ResNet-18 con pesos aleatorios (desde cero), 3 canales de entrada y 18 salidas (logits)."""
    import timm

    return timm.create_model("resnet18", pretrained=False, num_classes=NUM_TYPES, in_chans=3)


def to_unit_range(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1] (blanco = +1, igual que el GAN)."""
    return x_u8.float() / 127.5 - 1.0


def random_shift_flip(x: torch.Tensor, max_shift: int = 8) -> torch.Tensor:
    """Aumento por muestra: traslación entera en [-max_shift, max_shift] con relleno BLANCO (+1)
    y volteo horizontal con p = 0.5. x: float [B, C, H, W] en [-1, 1]."""
    B, C, H, W = x.shape
    s = max_shift
    padded = F.pad(x, (s, s, s, s), value=1.0)  # +1 = blanco
    dy = torch.randint(0, 2 * s + 1, (B,), device=x.device)
    dx = torch.randint(0, 2 * s + 1, (B,), device=x.device)
    rows = dy[:, None] + torch.arange(H, device=x.device)  # B, H
    cols = dx[:, None] + torch.arange(W, device=x.device)  # B, W
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(C, device=x.device)[None, :, None, None]
    out = padded[b, c, rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def views_as_samples(d: PairData) -> tuple[torch.Tensor, torch.Tensor]:
    """Frentes y espaldas como muestras independientes: (uint8 [2N, 3, 96, 96], float [2N, 18])."""
    imgs = np.concatenate([d.front, d.back], axis=0).transpose(0, 3, 1, 2)
    y = np.concatenate([d.y, d.y], axis=0).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(imgs)), torch.from_numpy(y)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x_u8: torch.Tensor, batch: int = 512) -> torch.Tensor:
    """Logits [N, 18] (en CPU) para imágenes uint8 [N, 3, 96, 96]. Deja el modelo en eval()."""
    model.eval()
    device = next(model.parameters()).device
    out = [model(to_unit_range(x_u8[i:i + batch].to(device))).float().cpu()
           for i in range(0, len(x_u8), batch)]
    return torch.cat(out) if out else torch.zeros(0, NUM_TYPES)


def tune_thresholds(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Umbral por clase que maximiza F1 en validación (búsqueda en una grilla 0.02..0.98).
    Clases sin positivos en validación quedan en 0.5."""
    grid = np.arange(0.02, 0.99, 0.02)
    th = np.full(probs.shape[1], 0.5, dtype=np.float32)
    for t in range(probs.shape[1]):
        pos = y_true[:, t] > 0.5
        if not pos.any():
            continue
        pred = probs[:, t][None, :] >= grid[:, None]          # G, N
        tp = (pred & pos).sum(1)
        f1 = 2 * tp / (pred.sum(1) + pos.sum() + 1e-12)
        th[t] = grid[int(np.argmax(f1))]
    return th


def train_type_classifier(
    train: PairData,
    val: PairData,
    out_dir: str | Path = "runs/type_classifier",
    epochs: int = 30,
    batch: int = 256,
    lr: float = 1e-3,
    wd: float = 0.05,
    device=None,
    patience: int = 8,
    seed: int = 0,
) -> dict:
    """Entrena el clasificador y guarda out_dir/best.pt = {'state_dict', 'thresholds', 'val_metrics'}.

    Devuelve las métricas de validación del mejor modelo (con umbrales ajustados en validación).
    """
    from .evaluate import type_metrics

    device = torch.device(device or ("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"))
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr = views_as_samples(train)
    x_va, y_va = views_as_samples(val)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)

    # pos_weight_t = clip((N - n_t) / n_t, 1, 10): pesa más los positivos de tipos raros.
    n_t = y_tr.sum(0).clamp_min(1)
    pos_weight = ((len(y_tr) - n_t) / n_t).clamp(1, 10)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = make_classifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(x_tr) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    best_map, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), device=device)
        total = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * batch:(i + 1) * batch]
            x = random_shift_flip(to_unit_range(x_tr[idx]))
            loss = loss_fn(model(x), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()

        val_map = type_metrics(predict_logits(model, x_va), y_va)["mAP"]
        print(f"[clasificador] época {epoch + 1}/{epochs}  pérdida {total / steps_per_epoch:.4f}  "
              f"mAP val {val_map:.4f}", flush=True)
        if val_map > best_map:
            best_map, bad_epochs = val_map, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[clasificador] parada temprana en la época {epoch + 1}", flush=True)
                break

    model.load_state_dict(best_state)
    probs = torch.sigmoid(predict_logits(model, x_va)).numpy()
    thresholds = tune_thresholds(probs, y_va.numpy())
    val_metrics = type_metrics(probs, y_va.numpy(), thresholds)
    torch.save({"state_dict": best_state, "thresholds": thresholds, "val_metrics": val_metrics},
               out_dir / "best.pt")
    return val_metrics


def load_classifier(path: str | Path, device=None) -> tuple[torch.nn.Module, np.ndarray]:
    """Carga best.pt -> (modelo en eval() en `device`, umbrales por clase np.float32[18])."""
    device = torch.device(device or ("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") == "probe":  # linear probe sobre backbone preentrenado (type_probe.py)
        from .type_probe import load_probe

        return load_probe(ckpt, device)
    model = make_classifier()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), np.asarray(ckpt["thresholds"], dtype=np.float32)
