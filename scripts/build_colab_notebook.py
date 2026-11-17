"""Genera notebooks/PokeGAN_Colab.ipynb: inferencia de PokeGAN en Google Colab (o Jupyter local).

Uso:
    python scripts/build_colab_notebook.py && python scripts/export_colab.py
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
TIPOS = ["normal", "lucha", "volador", "veneno", "tierra", "roca", "bicho", "fantasma", "acero", "fuego", "agua",
         "planta", "electrico", "psiquico", "hielo", "dragon", "siniestro", "hada"]
opts = ", ".join(f'"{t}"' for t in TIPOS)

cells = [
    md("""# PokeGAN — generación condicional de Pokémon (vista frontal y trasera)

GAN condicional entrenada desde cero. A partir de un único vector latente $z$ y del tipo (uno o dos de los 18),
el generador produce la vista **frontal** y la **trasera** de una criatura nueva, en 96×96 px.

| Modelo | Descripción |
|---|---|
| `pura` | Generador y discriminadores entrenados desde cero. |
| `vision` | Mismo generador (desde cero); durante el entrenamiento se añadió un discriminador auxiliar sobre características de ConvNeXt-T congelado. |

**Requisito:** el archivo `pokegan_colab.zip` (código y pesos). Se puede arrastrar al panel *Archivos* de Colab
antes de ejecutar la celda 1, o subirlo cuando la celda lo solicite."""),

    code("""#@title 1. Preparación
import os, sys, zipfile

ZIP = next((p for p in ["pokegan_colab.zip", "../colab/pokegan_colab.zip"] if os.path.exists(p)), None)
if ZIP is None:
    from google.colab import files
    files.upload()
    ZIP = "pokegan_colab.zip"

zipfile.ZipFile(ZIP).extractall("pokegan_colab")
sys.path.insert(0, "pokegan_colab")

from pokegan.demo import cargar_modelos, generar, variar_tipo
cargar_modelos("pokegan_colab/modelos");"""),

    md("""## 2. Generación

Cada columna es una criatura: arriba su vista frontal y abajo la trasera, ambas generadas desde el mismo $z$.
Modificar los valores y ejecutar la celda. La semilla fija las criaturas: la misma semilla reproduce el resultado."""),

    code(f"""#@title 2. Parámetros
TIPO_1   = "fuego"     #@param [{opts}]
TIPO_2   = "volador"   #@param ["ninguno", {opts}]
SEMILLA  = 0           #@param {{type:"integer"}}
CANTIDAD = 5           #@param {{type:"slider", min:1, max:10}}
MODELO   = "vision"    #@param ["vision", "pura"]

generar(TIPO_1, TIPO_2, semilla=SEMILLA, cantidad=CANTIDAD, modelo=MODELO);"""),

    md("""## 3. Varias combinaciones

Lista de pares `(tipo 1, tipo 2)`; `None` indica un solo tipo. El modelo también acepta combinaciones que no
aparecen en el conjunto de entrenamiento (por ejemplo fuego/hada)."""),

    code("""COMBINACIONES = [
    ("agua", None),
    ("planta", "veneno"),
    ("electrico", "acero"),
    ("fuego", "hada"),
]

for tipo_1, tipo_2 in COMBINACIONES:
    generar(tipo_1, tipo_2, semilla=SEMILLA, cantidad=CANTIDAD, modelo=MODELO);"""),

    md("""## 4. Efecto del tipo con $z$ fijo

Se mantiene el mismo $z$ y solo se cambia el tipo: $z$ determina la forma y el tipo determina principalmente la paleta."""),

    code("""variar_tipo(semilla=7, tipos=["fuego", "agua", "planta", "electrico", "hielo", "fantasma"], modelo=MODELO);"""),

    md("""## 5. Opciones adicionales

- `fidelidad=True`: truncamiento de los primeros bloques de $z$ (siluetas más limpias, menor variedad).
- `pixel_art=True`: post-procesamiento con paleta conjunta de 10 colores y limpieza del fondo."""),

    code("""generar("agua", None, semilla=SEMILLA, cantidad=CANTIDAD, modelo=MODELO, fidelidad=True, pixel_art=True);"""),

    md("""## 6. Código del modelo

El código fuente completo está en `pokegan_colab/pokegan/` (panel *Archivos*). Las piezas principales:"""),

    code("""import inspect
from pokegan.models import generator, discriminator
from pokegan import losses

print(inspect.getsource(generator.Generator.forward))
print(inspect.getsource(discriminator.Discriminator.forward))
print(inspect.getsource(losses.d_hinge))"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                  "colab": {"provenance": []}}
nbf.validate(nb)
out = ROOT / "notebooks" / "PokeGAN_Colab.ipynb"
nbf.write(nb, out)
print("Escrito", out)
