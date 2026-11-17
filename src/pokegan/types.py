"""Vocabulario de los 18 tipos y utilidades de codificación multi-hot."""

from __future__ import annotations

import numpy as np

# Orden canónico = orden de las columnas type_* de pairs.csv (= type_id 1..18).
TYPES: tuple[str, ...] = (
    "normal", "fighting", "flying", "poison", "ground", "rock",
    "bug", "ghost", "steel", "fire", "water", "grass",
    "electric", "psychic", "ice", "dragon", "dark", "fairy",
)
NUM_TYPES = len(TYPES)
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}

# Nombres en español para figuras e interfaz.
TYPES_ES = {
    "normal": "Normal", "fighting": "Lucha", "flying": "Volador", "poison": "Veneno",
    "ground": "Tierra", "rock": "Roca", "bug": "Bicho", "ghost": "Fantasma",
    "steel": "Acero", "fire": "Fuego", "water": "Agua", "grass": "Planta",
    "electric": "Eléctrico", "psychic": "Psíquico", "ice": "Hielo", "dragon": "Dragón",
    "dark": "Siniestro", "fairy": "Hada",
}
def _fold(s: str) -> str:
    """minúsculas y sin tildes: 'Eléctrico' -> 'electrico'."""
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", s.strip().lower()) if unicodedata.category(c) != "Mn")


_ES_TO_EN = {_fold(v): k for k, v in TYPES_ES.items()}


def normalize_type(name: str) -> str:
    """Acepta nombre en inglés o español, sin importar mayúsculas ni tildes ('electrico', 'Dragón')."""
    key = _fold(name)
    if key in TYPE_TO_IDX:
        return key
    if key in _ES_TO_EN:
        return _ES_TO_EN[key]
    raise ValueError(f"Tipo desconocido: {name!r}. Válidos: {', '.join(TYPES)}")


def encode(types: list[str] | tuple[str, ...]) -> np.ndarray:
    """['fire','flying'] -> vector multi-hot float32 de 18 posiciones (1 o 2 activas)."""
    if not 1 <= len(types) <= 2:
        raise ValueError("Se requiere 1 o 2 tipos")
    idx = {TYPE_TO_IDX[normalize_type(t)] for t in types}
    if len(idx) != len(types):
        raise ValueError("Tipos repetidos")
    y = np.zeros(NUM_TYPES, dtype=np.float32)
    y[list(idx)] = 1.0
    return y


def decode(y: np.ndarray) -> list[str]:
    return [TYPES[i] for i in np.flatnonzero(np.asarray(y) > 0.5)]


def label(y: np.ndarray, lang: str = "es") -> str:
    names = decode(y)
    if lang == "es":
        names = [TYPES_ES[n] for n in names]
    return "/".join(names)


def all_conditions() -> list[tuple[str, ...]]:
    """Las 18 + 153 = 171 condiciones posibles (monotipo y pares no ordenados)."""
    out: list[tuple[str, ...]] = [(t,) for t in TYPES]
    for i in range(NUM_TYPES):
        for j in range(i + 1, NUM_TYPES):
            out.append((TYPES[i], TYPES[j]))
    return out
