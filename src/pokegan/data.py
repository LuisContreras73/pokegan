"""Datos: caché uint8 de los pares frente/espalda, filtros, split por especie y pesos de muestreo.

Toda la colección (~12.8k pares x 2 vistas x 96x96x3 bytes ≈ 0.7 GB) cabe en RAM/VRAM como
uint8, así que evitamos DataLoader con workers (lento en Windows) y muestreamos índices
directamente sobre tensores en el dispositivo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import NUM_TYPES, TYPES

TYPE_COLS = [f"type_{t}" for t in TYPES]
GRAY_SAT_THRESHOLD = 0.08  # saturación media del primer plano bajo la cual un sprite es "gris"
WHITE_THRESHOLD = 0.97     # píxel con min(R,G,B) >= umbral se considera fondo blanco


def foreground_saturation(img: np.ndarray) -> float:
    """Saturación HSV media de los píxeles que no son fondo blanco. img: HxWx3 uint8."""
    a = img.astype(np.float32) / 255.0
    fg = a.min(-1) < WHITE_THRESHOLD
    if not fg.any():
        return 0.0
    mx, mn = a.max(-1), a.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    return float(sat[fg].mean())


def build_cache(csv_path: str | Path, root: str | Path, out_path: str | Path) -> dict:
    """Lee pairs.csv + PNGs y guarda un .npz con arrays uint8 y metadatos."""
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    root = Path(root)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    n = len(df)
    front = np.empty((n, 96, 96, 3), np.uint8)
    back = np.empty((n, 96, 96, 3), np.uint8)
    sat = np.empty((n, 2), np.float32)
    for i, (fp, bp) in enumerate(tqdm(zip(df.front_path, df.back_path), total=n, desc="cache")):
        f = np.asarray(Image.open(root / fp).convert("RGB"))
        b = np.asarray(Image.open(root / bp).convert("RGB"))
        assert f.shape == (96, 96, 3) and b.shape == (96, 96, 3), (fp, bp, f.shape, b.shape)
        front[i], back[i] = f, b
        sat[i] = foreground_saturation(f), foreground_saturation(b)

    y = df[TYPE_COLS].to_numpy(np.uint8)
    collections = sorted(df.source_collection.unique())
    meta = dict(
        y=y,
        species_id=df.species_id.to_numpy(np.int32),
        pokemon_form_id=df.pokemon_form_id.to_numpy(np.int32),
        is_shiny=df.is_shiny.to_numpy(np.uint8),
        generation=df.source_generation.to_numpy(np.int8),
        collection=np.array([collections.index(c) for c in df.source_collection], np.int16),
        collection_names=np.array(collections),
        species_name=df.species_name.to_numpy(str),
        pair_id=df.pair_id.to_numpy(str),
        saturation=sat,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, front=front, back=back, **meta)
    return {"n": n, "path": str(out_path)}


@dataclass
class PairData:
    front: np.ndarray        # N,96,96,3 uint8
    back: np.ndarray         # N,96,96,3 uint8
    y: np.ndarray            # N,18 float32 multi-hot
    species_id: np.ndarray   # N
    is_shiny: np.ndarray     # N
    collection: np.ndarray   # N
    collection_names: np.ndarray
    species_name: np.ndarray
    saturation: np.ndarray   # N,2

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "PairData":
        return PairData(
            front=self.front[mask], back=self.back[mask], y=self.y[mask],
            species_id=self.species_id[mask], is_shiny=self.is_shiny[mask],
            collection=self.collection[mask], collection_names=self.collection_names,
            species_name=self.species_name[mask], saturation=self.saturation[mask],
        )

    @property
    def is_gray(self) -> np.ndarray:
        return (self.saturation < GRAY_SAT_THRESHOLD).all(axis=1)


def load_cache(path: str | Path, drop_gray: bool = True, drop_shiny: bool = False) -> PairData:
    z = np.load(path, allow_pickle=False)
    d = PairData(
        front=z["front"], back=z["back"], y=z["y"].astype(np.float32),
        species_id=z["species_id"], is_shiny=z["is_shiny"], collection=z["collection"],
        collection_names=z["collection_names"], species_name=z["species_name"],
        saturation=z["saturation"],
    )
    keep = np.ones(len(d), bool)
    if drop_gray:
        keep &= ~d.is_gray
    if drop_shiny:
        keep &= d.is_shiny == 0
    return d.subset(keep)


def species_split(species_id: np.ndarray, val_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Máscara booleana 'es validación', asignando ESPECIES completas (evita fuga entre splits).

    Determinista: hash(seed, species) -> [0,1).
    """
    def u(s: int) -> float:
        h = hashlib.sha256(f"{seed}:{int(s)}".encode()).digest()
        return int.from_bytes(h[:8], "little") / 2**64

    uniq = np.unique(species_id)
    val_species = {s for s in uniq if u(s) < val_frac}
    return np.isin(species_id, list(val_species))


def combo_keys(y: np.ndarray) -> np.ndarray:
    """Codifica cada multi-hot como entero (bitmask) para agrupar combinaciones de tipos."""
    return (np.asarray(y) > 0.5).astype(np.int64) @ (1 << np.arange(NUM_TYPES, dtype=np.int64))


def sampling_weights(
    d: PairData, species_power: float = 0.5, combo_power: float = 0.25, shiny_factor: float = 0.5
) -> np.ndarray:
    """Pesos de muestreo que suavizan los desbalances del dataset.

    w_i ∝ n_pares(especie_i)^-species_power · n_especies(combo_i)^-combo_power · (shiny_factor si shiny)
      - especie: Unown/Pikachu aparecen cientos de veces; potencia 0.5 = 'raíz inversa'.
      - combo: se cuenta en ESPECIES distintas (no filas) para que un combo de una sola especie
        muy repetida no parezca frecuente.
      - shiny (~50 %): paletas alternativas que debilitan tipo→color; se atenúan, no se eliminan.
    Potencias 0 y shiny_factor 1 = muestreo uniforme por fila.
    """
    _, s_inv, s_cnt = np.unique(d.species_id, return_inverse=True, return_counts=True)
    keys = combo_keys(d.y)
    uk, c_inv = np.unique(keys, return_inverse=True)
    species_per_combo = np.array([len(np.unique(d.species_id[keys == k])) for k in uk])
    w = s_cnt[s_inv].astype(np.float64) ** -species_power
    w *= species_per_combo[c_inv].astype(np.float64) ** -combo_power
    w *= np.where(d.is_shiny == 1, shiny_factor, 1.0)
    return w / w.sum()


def load_curated(
    path: str | Path,
    val_frac: float = 0.05,
    seed: int = 0,
    exclude_prefixes: tuple[str, ...] = ("generation-1", "generation-2"),
) -> tuple[PairData, PairData]:
    """Subconjunto del diseño: sin colecciones GB/GBC (4 colores) y split por especie.

    El filtro de gris queda apagado: tras quitar Gen 1-2 los sprites grises restantes son
    Pokémon realmente grises (Roca/Acero) y quitarlos sesgaría esos tipos.
    """
    d = load_cache(path, drop_gray=False)
    bad = np.array([str(n).startswith(exclude_prefixes) for n in d.collection_names])
    d = d.subset(~bad[d.collection])
    val = species_split(d.species_id, val_frac, seed=seed)
    return d.subset(~val), d.subset(val)


def to_tensor_pairs(d: PairData, device="cpu"):
    """uint8 [N, 6, 96, 96] = [frente RGB; espalda RGB] en el dispositivo pedido."""
    import torch

    x = np.concatenate([d.front, d.back], axis=-1)  # N,96,96,6
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
