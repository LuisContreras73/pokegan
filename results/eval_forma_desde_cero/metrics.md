# Evaluación de PokeGAN

Checkpoint `runs/main/final.pt` (paso 27000, λ = 1.00); 2000 pares generados por semilla, semillas 0 (media ± desviación entre semillas). Inception se usa SOLO para evaluar.

## Calidad de distribución (Inception-v3, rasgos de 2048 dims)

| Métrica | Frente | Espalda |
|---|---|---|
| KID ×10³ generado vs. entrenamiento | 24.01 ± 0.00 | 41.83 ± 0.00 |
| FID generado vs. entrenamiento | 47.1 ± 0.0 | 69.1 ± 0.0 |
| Referencia: KID ×10³ real (validación) vs. entrenamiento | 3.07 | 2.44 |
| Referencia: FID real (validación) vs. entrenamiento* | 53.3 | 68.3 |

\* El FID está sesgado hacia arriba con N chico (validación ≈ 350 pares): no comparar FIDs con distinto N. El KID es insesgado y es la métrica titular.

## Condicionamiento por tipo (clasificador entrenado desde cero; protocolo GAN-test)

| | mAP | F1 macro | Acierto del conjunto |
|---|---|---|---|
| Generado | 0.213 ± 0.000 | 0.211 ± 0.000 | 0.152 ± 0.000 |
| Techo: reales de validación | 0.238 | 0.312 | 0.113 |
| Piso: condiciones barajadas | 0.084 ± 0.000 | 0.090 ± 0.000 | 0.033 ± 0.000 |
| Puntaje normalizado (gen − piso)/(techo − piso) | 0.84 | 0.54 | 1.50 |

Acierto del conjunto = los |c| tipos con mayor puntaje coinciden con la condición. Los umbrales de F1 se ajustaron en validación (el techo de F1 es algo optimista).

| Tipo | Recall generado | Recall real (validación) |
|---|---|---|
| Normal | 0.017 | 0.212 |
| Lucha | 0.179 | 0.477 |
| Volador | 0.126 | 0.618 |
| Veneno | 0.298 | 0.333 |
| Tierra | 0.725 | 0.500 |
| Roca | 0.175 | 0.173 |
| Bicho | 0.618 | 0.614 |
| Fantasma | 0.143 | 0.278 |
| Acero | 0.108 | 0.667 |
| Fuego | 0.607 | 0.447 |
| Agua | 0.903 | 0.776 |
| Planta | 0.351 | 0.292 |
| Eléctrico | 0.217 | 0.167 |
| Psíquico | 0.981 | 0.933 |
| Hielo | 0.012 | 0.250 |
| Dragón | 0.033 | 0.154 |
| Siniestro | 0.449 | 0.528 |
| Hada | 0.210 | 0.250 |

## Coherencia frente/espalda (intersección de histogramas de paleta, 8×8×8 colores)

| | HI de paleta |
|---|---|
| Generado | 0.599 ± 0.000 |
| Real emparejado (techo) | 0.634 |
| Real barajado, espalda de otra especie (piso) | 0.109 |
| Índice normalizado (gen − piso)/(techo − piso) | 0.93 |

## Diversidad dentro de una condición (18 monotipos)

| | Distancia coseno (Inception, par) | Vendi (criaturas efectivas) | L1 en píxeles |
|---|---|---|---|
| Generado | 0.271 | 3.7 | 0.077 |
| Real (un sprite por especie) | 0.301 | 4.2 | 0.129 |
| Razón generado / real | 0.90 | 0.88 | 0.60 |

Por tipo se comparan grupos del mismo tamaño (tantas muestras generadas como especies reales monotipo, máx. 50), porque Vendi crece con el número de muestras. Razón cercana a 1 = diversidad como la real; < 0.5 sugiere colapso parcial de modos.

## Novedad (vecino más cercano del entrenamiento, L2 en píxeles, frentes)

- Fracción de generados más cerca del entrenamiento que el 5 % más cercano de los reales de validación (especies nunca vistas): **0.377** (≈ 0.05 si no hay copia).
- Mediana de la distancia al vecino: generados 24.6 vs. validación 32.4; Mann-Whitney (gen < val) p = 0.
- Figuras: `nn_closest.png` (los más parecidos) y `nn_random.png` (al azar).

## Combinaciones de tipos nunca vistas en entrenamiento

Tasa en que AMBOS tipos de la condición quedan en el top-2 del clasificador.

| Combinación no vista | Tasa |
|---|---|
| Normal/Roca | 0.007 |
| Normal/Bicho | 0.000 |
| Normal/Acero | 0.000 |
| Normal/Hielo | 0.000 |
| Veneno/Hielo | 0.000 |
| Tierra/Hada | 0.028 |
| Roca/Fantasma | 0.135 |
| Bicho/Dragón | 0.000 |
| Bicho/Siniestro | 0.030 |
| Fuego/Hada | 0.030 |
| Dragón/Hada | 0.005 |
| **Media no vistas** | **0.021** |
| Media de 11 combinaciones vistas frecuentes (referencia) | 0.047 |

Filas de entrenamiento con alguna combinación "no vista": 0 (debe ser 0).
