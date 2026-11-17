# Evaluación de PokeGAN

Checkpoint `runs/main/final.pt` (paso 27000, λ = 1.00); 3000 pares generados por semilla, semillas 0, 1 (media ± desviación entre semillas). Inception se usa SOLO para evaluar.

## Calidad de distribución (Inception-v3, rasgos de 2048 dims)

| Métrica | Frente | Espalda |
|---|---|---|
| KID ×10³ generado vs. entrenamiento | 24.03 ± 0.20 | 42.75 ± 0.47 |
| FID generado vs. entrenamiento | 44.9 ± 0.7 | 67.7 ± 0.4 |
| Referencia: KID ×10³ real (validación) vs. entrenamiento | 3.07 | 2.44 |
| Referencia: FID real (validación) vs. entrenamiento* | 53.3 | 68.3 |

\* El FID está sesgado hacia arriba con N chico (validación ≈ 350 pares): no comparar FIDs con distinto N. El KID es insesgado y es la métrica titular.

## Condicionamiento por tipo (linear probe sobre DINOv3 congelado (preentrenado, SOLO evaluación); protocolo GAN-test)

| | mAP | F1 macro | Acierto del conjunto |
|---|---|---|---|
| Generado | 0.214 ± 0.000 | 0.201 ± 0.000 | 0.170 ± 0.001 |
| Techo: reales de validación | 0.347 | 0.406 | 0.228 |
| Piso: condiciones barajadas | 0.084 ± 0.001 | 0.077 ± 0.002 | 0.047 ± 0.001 |
| Puntaje normalizado (gen − piso)/(techo − piso) | 0.49 | 0.38 | 0.68 |

Acierto del conjunto = los |c| tipos con mayor puntaje coinciden con la condición. Los umbrales de F1 se ajustaron en validación (el techo de F1 es algo optimista).

| Tipo | Recall generado | Recall real (validación) |
|---|---|---|
| Normal | 0.053 | 0.550 |
| Lucha | 0.002 | 0.534 |
| Volador | 0.017 | 0.632 |
| Veneno | 0.631 | 0.812 |
| Tierra | 0.594 | 0.472 |
| Roca | 0.565 | 0.288 |
| Bicho | 0.118 | 0.614 |
| Fantasma | 0.167 | 0.333 |
| Acero | 0.003 | 0.167 |
| Fuego | 0.689 | 0.645 |
| Agua | 0.517 | 0.534 |
| Planta | 0.236 | 0.479 |
| Eléctrico | 0.124 | 0.250 |
| Psíquico | 0.329 | 0.525 |
| Hielo | 0.418 | 0.250 |
| Dragón | 0.000 | 0.519 |
| Siniestro | 0.566 | 0.611 |
| Hada | 0.336 | 0.523 |

## Coherencia frente/espalda (intersección de histogramas de paleta, 8×8×8 colores)

| | HI de paleta |
|---|---|
| Generado | 0.600 ± 0.001 |
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
| Normal/Hielo | 0.005 |
| Veneno/Hielo | 0.115 |
| Tierra/Hada | 0.022 |
| Roca/Fantasma | 0.020 |
| Bicho/Dragón | 0.000 |
| Bicho/Siniestro | 0.003 |
| Fuego/Hada | 0.060 |
| Dragón/Hada | 0.000 |
| **Media no vistas** | **0.021** |
| Media de 11 combinaciones vistas frecuentes (referencia) | 0.073 |

Filas de entrenamiento con alguna combinación "no vista": 0 (debe ser 0).
