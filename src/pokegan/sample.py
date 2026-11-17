"""Muestreo con el generador entrenado: cargar checkpoint, generar pares frente/espalda e interpolar.

Convenciones:
  - G(z, y, lam) devuelve [B, 6, 96, 96] en [-1, 1] (canales 0:3 = frente, 3:6 = espalda; blanco = +1).
  - Aquí todo se devuelve como numpy uint8 [n, 2, 96, 96, 3] (eje 1: 0 = frente, 1 = espalda).
  - z se muestrea en CPU con un torch.Generator propio: misma semilla => mismos Pokémon.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import types as T


def default_device(device=None) -> torch.device:
    """El dispositivo pedido o, si no se pide ninguno, cuda cuando está disponible (si no, cpu)."""
    if device is None:
        # device_count(): con CUDA_VISIBLE_DEVICES vacío, is_available() puede dar True sin GPUs.
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    return torch.device(device)


def model_device(G) -> torch.device:
    """Dispositivo donde viven los pesos de G (cpu si no tiene pesos)."""
    t = next(G.parameters(), None)
    if t is None:
        t = next(G.buffers(), None)
    return t.device if t is not None else torch.device("cpu")


def load_generator(ckpt_path, ema: bool = True, device=None, standing: bool = True,
                   standing_seed: int = 0):
    """Carga G desde un checkpoint y lo deja listo para muestrear (modo eval).

    ema=True usa los pesos promediados G_ema (mejores muestras); ema=False usa G crudo.
    standing=True recalcula las estadísticas de BatchNorm ("standing stats") con condiciones del
    banco de etiquetas de entrenamiento y con el lam guardado: las medias móviles de BN no
    corresponden a los pesos promediados de G_ema y, sin esto, las muestras salen "lavadas".
    Devuelve (G, info) con info = {'step', 'lam', 'cfg', 'label_bank'}.
    """
    # Import tardío: así este módulo se puede importar (y probar) sin los modelos.
    from .models.generator import Generator

    device = default_device(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G = Generator(**ckpt["cfg"]["G"]).to(device)
    if not ema and "G" not in ckpt:
        raise ValueError(f"{ckpt_path} solo trae G_ema (snapshot/best): usa ema=True")
    G.load_state_dict(ckpt["G_ema" if ema else "G"])
    G.eval()
    lam = float(ckpt.get("lam", 1.0))
    bank = ckpt.get("label_bank")
    if bank is not None:
        bank = torch.as_tensor(bank).float().cpu()
    # BN debe ver la misma mezcla de condiciones que en el entrenamiento (muestreo ponderado)
    bn_bank = ckpt.get("bn_label_bank", bank)
    # los pesos publicados (scripts/export_weights.py) ya traen BatchNorm recalculada
    if standing and bn_bank is not None and not ckpt.get("standing_applied", False):
        from .ema import standing_stats

        # standing_stats usa z aleatorios: fijamos la semilla (sin alterar el RNG global)
        # para que dos ejecuciones con la misma semilla den exactamente las mismas imágenes.
        cuda_ids = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_ids):
            torch.manual_seed(standing_seed)
            standing_stats(G, torch.as_tensor(bn_bank).float().to(device), lam=lam)
        G.eval()
    info = {"step": int(ckpt.get("step", 0)), "lam": lam, "cfg": ckpt["cfg"], "label_bank": bank}
    return G, info


def parse_types(types) -> list[str]:
    """'fuego/volador', 'fuego volador' o ['Fuego', 'flying'] -> lista de nombres de tipo."""
    if isinstance(types, str):
        types = types.replace("/", " ").replace(",", " ").split()
    return list(types)


def cond_label(types, lang: str = "es") -> str:
    """Etiqueta legible de una condición: ['fire', 'flying'] -> 'Volador/Fuego'."""
    return T.label(T.encode(parse_types(types)), lang)


def make_z(n: int, z_dim: int, seed: int = 0, truncation: float | None = None) -> torch.Tensor:
    """z ~ N(0, I) determinista (en CPU).

    Truco de truncamiento (BigGAN): cada componente con |z| > truncation se vuelve a muestrear.
    Valores pequeños (p. ej. 0.5) dan Pokémon más "típicos" y nítidos pero menos diversos.
    """
    g = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n, z_dim, generator=g)
    if truncation is not None:
        if truncation <= 0:
            raise ValueError("truncation debe ser > 0")
        bad = z.abs() > truncation
        while bad.any():
            z[bad] = torch.randn(int(bad.sum()), generator=g)
            bad = z.abs() > truncation
    return z


def make_y(types, n: int, seed: int = 0, label_bank=None) -> torch.Tensor:
    """Condiciones multi-hot [n, 18].

    types dado (es/en) => la misma condición n veces. types=None => condiciones aleatorias:
    filas del banco de etiquetas de entrenamiento (misma distribución que los datos) o, si no
    hay banco, una de las 171 combinaciones posibles al azar.
    """
    if types is not None:
        return torch.from_numpy(T.encode(parse_types(types))).repeat(n, 1)
    g = torch.Generator().manual_seed(int(seed))
    if label_bank is not None:
        bank = torch.as_tensor(label_bank).float().cpu()
        return bank[torch.randint(len(bank), (n,), generator=g)]
    conds = T.all_conditions()
    idx = torch.randint(len(conds), (n,), generator=g).tolist()
    return torch.from_numpy(np.stack([T.encode(conds[i]) for i in idx]))


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[B, 6, H, W] en [-1, 1] -> numpy uint8 [B, 2, H, W, 3] (frente, espalda)."""
    u8 = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()
    B, _, H, W = u8.shape
    return u8.reshape(B, 2, 3, H, W).permute(0, 1, 3, 4, 2).contiguous().numpy()


@torch.no_grad()
def render(G, z: torch.Tensor, y: torch.Tensor, lam: float = 1.0, device=None,
           batch: int = 64) -> np.ndarray:
    """Pasa (z, y) por G en lotes (modo eval) y devuelve uint8 [n, 2, 96, 96, 3]."""
    G.eval()  # BN con estadísticas fijas: cada muestra no depende del resto del lote
    device = model_device(G) if device is None else torch.device(device)
    out = [G(z[i:i + batch].to(device), y[i:i + batch].to(device), lam).float().cpu()
           for i in range(0, len(z), batch)]
    return to_uint8(torch.cat(out))


def generate(G, types, n: int, seed: int = 0, truncation: float | None = None, lam: float = 1.0,
             label_bank=None, device=None) -> np.ndarray:
    """Genera n Pokémon nuevos -> uint8 [n, 2, 96, 96, 3] (eje 1: 0 = frente, 1 = espalda).

    types: 1 o 2 tipos en español o inglés (['fuego', 'volador'], 'Hada', ['fire', 'fairy'])
    o None para condiciones aleatorias (del label_bank si se entrega).
    La misma semilla da los mismos z: con otro tipo y la misma semilla se obtiene "el mismo
    individuo" con otra condición.
    """
    z = make_z(n, G.z_dim, seed, truncation)
    y = make_y(types, n, seed, label_bank)
    return render(G, z, y, lam, device)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Interpolación esférica entre dos z.

    En alta dimensión casi toda la masa de N(0, I) está cerca de la esfera de radio sqrt(d);
    la recta entre dos z pasa por normas más pequeñas (zona poco probable), el arco no.
    """
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
    omega = math.acos(max(-1.0, min(1.0, cos)))
    if omega < 1e-4:  # z casi iguales: basta la recta
        return (1 - t) * a + t * b
    return (math.sin((1 - t) * omega) * a + math.sin(t * omega) * b) / math.sin(omega)


def interpolate(G, types_a, types_b, z_a_seed: int, z_b_seed: int, steps: int = 8,
                lam: float = 1.0, truncation: float | None = None, device=None) -> np.ndarray:
    """Interpola a la vez z (slerp entre dos semillas) y la condición (mezcla lineal del multi-hot).

    - Tipo fijo, z variable: mismos tipos y semillas distintas -> cambia el individuo.
    - z fijo, tipo variable: misma semilla y tipos distintos (Fuego -> Agua) -> cambia el tipo.
    Devuelve uint8 [steps, 2, 96, 96, 3]; el paso k corresponde a t = k / (steps - 1).
    """
    za = make_z(1, G.z_dim, z_a_seed, truncation)[0]
    zb = make_z(1, G.z_dim, z_b_seed, truncation)[0]
    ya, yb = make_y(types_a, 1)[0], make_y(types_b, 1)[0]
    ts = np.linspace(0.0, 1.0, steps)
    z = torch.stack([slerp(za, zb, float(t)) for t in ts])
    y = torch.stack([(1 - float(t)) * ya + float(t) * yb for t in ts])
    return render(G, z, y, lam, device)


def pair_canvas(pair: np.ndarray) -> np.ndarray:
    """[2, H, W, 3] -> [H, 2W, 3]: frente | espalda lado a lado."""
    return np.concatenate([pair[0], pair[1]], axis=1)


def upscale(img: np.ndarray, k: int) -> np.ndarray:
    """Amplía por un entero k con vecino más cercano (conserva el pixel art)."""
    return img if k == 1 else np.repeat(np.repeat(img, k, axis=0), k, axis=1)
