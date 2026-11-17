"""PokeGAN: GAN condicional (tipos) que genera vistas frontal y trasera coherentes de Pokémon nuevos."""

import sys as _sys

__version__ = "0.1.0"

# La consola de Windows usa cp1252 por defecto y no puede imprimir λ, →, σ, ±… (UnicodeEncodeError).
for _s in (_sys.stdout, _sys.stderr):
    try:
        if _s is not None and (_s.encoding or "").lower().replace("-", "") != "utf8":
            _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
