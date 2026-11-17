"""Exporta los generadores a ONNX para la demo web estática (GitHub Pages + ONNX Runtime Web).

Crea web/models/<clave>.onnx y web/models/models.json:
  entradas  z [N, 120] float32 ~ N(0, I),  y [N, 18] float32 multi-hot
  salida    x [N, 6, 96, 96] en [-1, 1]  (canales 0-2 = frente, 3-5 = espalda)
λ y standing stats ya aplicados; la normalización espectral se hornea en los pesos. Los pesos se guardan en fp16
con un Cast a fp32 al cargar (la red se ejecuta en fp32), así cada modelo pesa ~17 MB.

Uso:
    python scripts/export_web.py
"""

from __future__ import annotations

import json
import types
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokegan import types as T  # noqa: E402
from pokegan.sample import load_generator  # noqa: E402

MODELS = {
    "vision": ("Vision", ROOT / "weights" / "pokegan_vision.pt"),
    "pura": ("Pura", ROOT / "weights" / "pokegan_pura.pt"),
}
OUT = ROOT / "web" / "models"


def _type_embedding_export(self, y: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Igual que Generator.type_embedding, pero (e·r)·λ en vez de λ·e·r: con λ junto al MatMul, el optimizador
    MatMulScaleFusion de ONNX Runtime fusiona λ y deja colgando la otra rama que usa e (modelo inválido)."""
    e = self.embed(y)
    return (e * torch.rsqrt(e.pow(2).mean(1, keepdim=True) + 1e-8)) * lam


class Exportable(torch.nn.Module):
    def __init__(self, G: torch.nn.Module, lam: float):
        super().__init__()
        G.type_embedding = types.MethodType(_type_embedding_export, G)
        self.G, self.lam = G, float(lam)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.G(z, y, lam=self.lam)


def bake_spectral_norm(G: torch.nn.Module) -> None:
    for m in list(G.modules()):
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


def weights_to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Initializers fp32 → fp16 + nodo Cast(→fp32): mitad de descarga, misma aritmética."""
    g = model.graph
    graph_inputs = {i.name for i in g.input}
    casts, keep, new = [], [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 64 and init.name not in graph_inputs:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new.append(numpy_helper.from_array(arr, init.name + "__fp16"))
            casts.append(helper.make_node("Cast", [init.name + "__fp16"], [init.name], to=TensorProto.FLOAT))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep + new)
    nodes = toposort(casts + list(g.node), {i.name for i in g.input} | {i.name for i in g.initializer})
    del g.node[:]
    g.node.extend(nodes)
    return model


def toposort(nodes: list, available: set) -> list:
    """Orden topológico (el exportador TorchScript de torch 2.11 puede emitir nodos fuera de orden)."""
    avail, pending, out = set(available) | {""}, list(nodes), []
    while pending:
        ready = [n for n in pending if all(i in avail for i in n.input)]
        if not ready:
            raise RuntimeError(f"grafo con dependencias sin resolver: {[n.name for n in pending[:5]]}")
        ids = {id(n) for n in ready}
        pending = [n for n in pending if id(n) not in ids]
        for n in ready:
            out.append(n)
            avail.update(n.output)
    return out


def make_logo(seed: int = 2024, index: int = 15) -> None:
    """Logotipo de la web: una criatura Fuego/Volador generada por el propio modelo vision (no es un personaje existente).

    Recorta el primer plano (fondo blanco conectado al borde → transparente) y guarda web/assets/logo.png.
    """
    from PIL import Image

    from pokegan.sample import make_y, make_z, render

    G, info = load_generator(ROOT / "weights" / "pokegan_vision.pt", device="cpu")
    z, y = make_z(index + 1, 120, seed), make_y(["fire", "flying"], index + 1, seed)
    img = render(G, z, y, lam=info["lam"], device="cpu")[index, 0]          # vista frontal, uint8 [96, 96, 3]
    white = img.min(axis=2) >= 235
    bg = np.zeros_like(white)
    stack = [(r, c) for r in range(96) for c in (0, 95)] + [(r, c) for c in range(96) for r in (0, 95)]
    while stack:
        r, c = stack.pop()
        if 0 <= r < 96 and 0 <= c < 96 and white[r, c] and not bg[r, c]:
            bg[r, c] = True
            stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    rows, cols = np.where(~bg)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    side = max(r1 - r0, c1 - c0) + 2
    rgba = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - (r1 - r0)) // 2, (side - (c1 - c0)) // 2
    rgba[oy:oy + r1 - r0, ox:ox + c1 - c0, :3] = img[r0:r1, c0:c1]
    rgba[oy:oy + r1 - r0, ox:ox + c1 - c0, 3] = np.where(bg[r0:r1, c0:c1], 0, 255)
    (ROOT / "web" / "assets").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(ROOT / "web" / "assets" / "logo.png")
    print(f"  logo: web/assets/logo.png ({side}×{side} px, semilla {seed}, muestra {index})")


def main() -> None:
    import argparse

    import onnxruntime as ort

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-logo", action="store_true", help="regenera solo web/assets/logo.png")
    a = ap.parse_args()
    torch.set_num_threads(4)
    make_logo()
    if a.solo_logo:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"z_dim": 120, "n_chunks": 6, "types": [[t, T.TYPES_ES[t]] for t in T.TYPES], "models": []}
    for key, (label, path) in MODELS.items():
        if not path.exists():
            print(f"  (sin {path.name}, se omite {key})")
            continue
        G, info = load_generator(path, ema=True, device="cpu", standing=True, standing_seed=0)
        G.eval()
        bake_spectral_norm(G)
        m = Exportable(G, info["lam"]).eval()
        z = torch.randn(4, 120, generator=torch.Generator().manual_seed(0))
        y = torch.from_numpy(np.stack([T.encode(["fire", "flying"])] * 2 + [T.encode(["water"])] * 2)).float()
        with torch.no_grad():
            ref = m(z, y).numpy()
        tmp = OUT / f"{key}_fp32.onnx"
        torch.onnx.export(m, (z, y), str(tmp), input_names=["z", "y"], output_names=["x"],
                          dynamic_axes={"z": {0: "n"}, "y": {0: "n"}, "x": {0: "n"}}, opset_version=17, dynamo=False)
        model = weights_to_fp16(onnx.load(str(tmp)))
        onnx.checker.check_model(model)
        dst = OUT / f"{key}.onnx"
        onnx.save(model, str(dst))
        tmp.unlink()
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        out = sess.run(["x"], {"z": z.numpy(), "y": y.numpy()})[0]
        diff = np.abs(out - ref)
        px = np.abs(np.round((out + 1) * 127.5) - np.round((ref + 1) * 127.5))
        print(f"  {key}: {dst.stat().st_size / 2**20:.1f} MB · |Δ| máx {diff.max():.4f} · Δ píxel medio {px.mean():.3f}/255")
        meta["models"].append({"key": key, "label": label, "step": info["step"], "file": f"models/{key}.onnx",
                               "mb": round(dst.stat().st_size / 2**20, 1)})
    (OUT / "models.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Listo:", OUT)


if __name__ == "__main__":
    main()
