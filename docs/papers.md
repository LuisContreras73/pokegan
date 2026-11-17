# Papers consultados

Los trabajos citados en [`matematica.md`](matematica.md) y consultados durante el diseño, agrupados por tema. Los identificadores de arXiv se verificaron contra la API de arXiv durante la investigación.

- **Año**: año de la primera versión en arXiv (el que indica el identificador). Algunos se publicaron en congresos
  al año siguiente; por ejemplo, R3GAN aparece en NeurIPS 2024.
- **Archivo local**: ruta dentro del repositorio si el PDF se descargó; «—» si no. **Los PDF no se suben a GitHub**
  (`papers/**/*.pdf` está en `.gitignore`), así que las rutas solo existen en la copia local. Abajo se explica cómo
  volver a descargarlos.
- ⭐ marca los papers en que se apoya directamente una pieza de nuestro modelo.

## 1. Fundamentos de GANs y pérdidas

| Título | arXiv | Año | Archivo local | Por qué importa |
|---|---|---|---|---|
| ⭐ Generative Adversarial Networks | [1406.2661](https://arxiv.org/abs/1406.2661) | 2014 | — | Juego minimax, $D^\star$ y reducción a $2\,\mathrm{JSD}-\log 4$ (§1 de la matemática). |
| ⭐ Towards Principled Methods for Training Generative Adversarial Networks | [1701.04862](https://arxiv.org/abs/1701.04862) | 2017 | — | Gradientes que se desvanecen con soportes disjuntos; la pérdida no saturante equivale a KL inversa − 2 JSD. |
| f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization | [1606.00709](https://arxiv.org/abs/1606.00709) | 2016 | `papers/loss_math/1606.00709_f-gan.pdf` | Unifica las pérdidas como $f$-divergencias: el discriminador estima una función del cociente de densidades. |
| Least Squares Generative Adversarial Networks | [1611.04076](https://arxiv.org/abs/1611.04076) | 2016 | — | LSGAN minimiza la discriminación triangular, la misma divergencia que aparece en LeCam. |
| Wasserstein GAN | [1701.07875](https://arxiv.org/abs/1701.07875) | 2017 | — | Distancia $W_1$ y dualidad de Kantorovich–Rubinstein: por qué importa la restricción Lipschitz. |
| Improved Training of Wasserstein GANs | [1704.00028](https://arxiv.org/abs/1704.00028) | 2017 | — | Penalización de gradiente centrada en 1; contraste con SN y con R1. |
| ⭐ Geometric GAN | [1705.02894](https://arxiv.org/abs/1705.02894) | 2017 | — | Origen de la pérdida hinge e interpretación como SVM de margen 1. |
| ⭐ Which Training Methods for GANs do actually Converge? | [1801.04406](https://arxiv.org/abs/1801.04406) | 2018 | `papers/loss_math/1801.04406_mescheder-which-gans-converge.pdf` | Dirac-GAN: la dinámica gira; R1/R2 como fricción y convergencia local. |
| Stabilizing Training of Generative Adversarial Networks through Regularization | [1705.09367](https://arxiv.org/abs/1705.09367) | 2017 | — | La penalización de gradiente aproxima ruido de instancia (vínculo con R1/R2). |
| The relativistic discriminator: a key element missing from standard GAN | [1807.00734](https://arxiv.org/abs/1807.00734) | 2018 | `papers/loss_math/1807.00734_relativistic-gan.pdf` | RpGAN/RaGAN: base de R3GAN (alternativa descartada hoy). |
| On Relativistic f-Divergences | [1901.02474](https://arxiv.org/abs/1901.02474) | 2019 | — | El objetivo relativista es una divergencia. |
| Towards a Better Global Loss Landscape of GANs | [2011.04926](https://arxiv.org/abs/2011.04926) | 2020 | — | La GAN separable tiene muchas «cuencas malas» (colapso); RpGAN no tiene ninguna. |
| The GAN is dead; long live the GAN! A Modern GAN Baseline (R3GAN) | [2501.05441](https://arxiv.org/abs/2501.05441) | 2025 | `papers/loss_math/2501.05441_r3gan-modern-baseline.pdf` | RpGAN + R1 + R2 con prueba de convergencia local; nuestro principal «trabajo futuro». |
| A Large-Scale Study on Regularization and Normalization in GANs | [1807.04720](https://arxiv.org/abs/1807.04720) | 2018 | — | SN necesita menos ajuste de hiperparámetros que la penalización de gradiente. |
| Are GANs Created Equal? A Large-Scale Study | [1711.10337](https://arxiv.org/abs/1711.10337) | 2017 | — | Con ajuste suficiente, ninguna variante supera sistemáticamente a la GAN no saturante. |
| Mixed Precision Training | [1710.03740](https://arxiv.org/abs/1710.03740) | 2017 | — | Trampas de fp16 (escalado, épsilon); por qué entrenamos en fp32 + TF32. |

## 2. Arquitectura y optimización

| Título | arXiv | Año | Archivo local | Por qué importa |
|---|---|---|---|---|
| ⭐ Spectral Normalization for Generative Adversarial Networks | [1802.05957](https://arxiv.org/abs/1802.05957) | 2018 | `papers/loss_math/1802.05957_spectral-normalization.pdf` | Cota de Lipschitz por $\prod\sigma(W_l)$ e iteración de potencia (§3). |
| The Singular Values of Convolutional Layers | [1805.10408](https://arxiv.org/abs/1805.10408) | 2018 | — | Norma exacta de una convolución: la SN por remodelado no es exactamente 1-Lipschitz. |
| ⭐ Large Scale GAN Training for High Fidelity Natural Image Synthesis (BigGAN) | [1809.11096](https://arxiv.org/abs/1809.11096) | 2018 | `papers/architecture/1809.11096_BigGAN.pdf` | Esqueleto de G: CBN, z jerárquico, embedding compartido, EMA, standing stats, init ortogonal. |
| ⭐ Self-Attention Generative Adversarial Networks (SAGAN) | [1805.08318](https://arxiv.org/abs/1805.08318) | 2018 | `papers/architecture/1805.08318_SAGAN.pdf` | Autoatención a resolución media (nuestra 24×24), SN también en G y TTUR 4e-4/1e-4. |
| ⭐ Modulating early visual processing by language (BatchNorm condicional) | [1707.00683](https://arxiv.org/abs/1707.00683) | 2017 | — | Origen de la BatchNorm condicional que usa G. |
| ⭐ GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium | [1706.08500](https://arxiv.org/abs/1706.08500) | 2017 | `papers/evaluation/1706.08500_FID_Heusel_TTUR.pdf` | Justifica tasas de aprendizaje distintas para G y D (TTUR) y define el FID. |
| ⭐ The Unusual Effectiveness of Averaging in GAN Training | [1806.04498](https://arxiv.org/abs/1806.04498) | 2018 | — | Promediar los pesos de G (EMA) mejora las muestras. |

## 3. Condicionamiento por tipos

| Título | arXiv | Año | Archivo local | Por qué importa |
|---|---|---|---|---|
| ⭐ cGANs with Projection Discriminator | [1802.05637](https://arxiv.org/abs/1802.05637) | 2018 | `papers/conditioning/1802.05637_projection-discriminator.pdf` | Deriva $C=\psi(\phi)+y^\top V\phi$ del log-cociente óptimo; base de nuestra extensión multi-hot por suma. |
| ⭐ Multimodal Conditional Image Synthesis with Product-of-Experts GANs (PoE-GAN) | [2112.05130](https://arxiv.org/abs/2112.05130) | 2021 | `papers/conditioning/2112.05130_poe-gan-multimodal-projection.pdf` | Con varias condiciones, el término de proyección es una suma por condición: precedente publicado de la suma. |
| Compositional Visual Generation and Inference with Energy Based Models | [2004.06030](https://arxiv.org/abs/2004.06030) | 2020 | — | Conjunción de conceptos como suma de energías (producto de expertos). |
| Conditional Image Synthesis With Auxiliary Classifier GANs (AC-GAN) | [1610.09585](https://arxiv.org/abs/1610.09585) | 2016 | — | Línea base con clasificador auxiliar, contra la que argumentamos. |
| ⭐ Twin Auxiliary Classifiers GAN (TAC-GAN) | [1907.02690](https://arxiv.org/abs/1907.02690) | 2019 | — | AC-GAN minimiza $H_Q(Y\mid X)$: muestras estereotipadas, poca diversidad. |
| Rebooting ACGAN: Auxiliary Classifier GANs with Stable Training (ReACGAN) | [2111.01118](https://arxiv.org/abs/2111.01118) | 2021 | `papers/conditioning/2111.01118_reacgan.pdf` | En CUB200 (pocos datos) AC-GAN colapsa y la proyección funciona bien. |
| StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis | [2206.09479](https://arxiv.org/abs/2206.09479) | 2022 | `papers/conditioning/2206.09479_studiogan-benchmark.pdf` | Benchmark justo de condicionamientos y configuraciones probadas (LeCam 0.3, EMA, lr). |
| ContraGAN: Contrastive Learning for Conditional Image Generation | [2006.12681](https://arxiv.org/abs/2006.12681) | 2020 | — | Pérdida contrastiva condicional; StudioGAN muestra que condiciona mal. |
| Conditional GANs with Auxiliary Discriminative Classifier (ADC-GAN) | [2107.10060](https://arxiv.org/abs/2107.10060) | 2021 | — | Clasificador que distingue clase en reales y falsos: alternativa segura si se necesitara. |
| Dual Projection Generative Adversarial Networks (P2GAN) | [2108.09016](https://arxiv.org/abs/2108.09016) | 2021 | — | Proyección como *data matching* frente a clasificador como *label matching*. |
| Omni-GAN: On the Secrets of cGANs and Beyond | [2011.13074](https://arxiv.org/abs/2011.13074) | 2020 | — | Pérdida multi-etiqueta y papel del weight decay en D según el tamaño del dataset. |
| StarGAN: Unified GANs for Multi-Domain Image-to-Image Translation | [1711.09020](https://arxiv.org/abs/1711.09020) | 2017 | — | Precedente de clasificador auxiliar multi-etiqueta con BCE por atributo. |
| ⭐ Collapse by Conditioning: Training Class-conditional GANs with Limited Data | [2201.06578](https://arxiv.org/abs/2201.06578) | 2022 | `papers/conditioning/2201.06578_collapse-by-conditioning.pdf` | Condicionar con pocos datos causa colapso; entrenamiento transicional $\lambda_t$ (§8). |
| NoisyTwins: Class-Consistent and Diverse Image Generation through StyleGANs | [2304.05866](https://arxiv.org/abs/2304.05866) | 2023 | — | Ruido en el embedding de clase contra el colapso en clases raras. |
| Taming the Tail in Class-Conditional GANs (UTLO) | [2402.17065](https://arxiv.org/abs/2402.17065) | 2024 | — | Capas de baja resolución incondicionales para compartir conocimiento con clases raras. |
| Improving GANs for Long-Tailed Data through Group Spectral Regularization | [2208.09932](https://arxiv.org/abs/2208.09932) | 2022 | — | El colapso por clase se asocia a la explosión espectral de los parámetros de la CBN. |
| Class-Balanced Loss Based on Effective Number of Samples | [1901.05555](https://arxiv.org/abs/1901.05555) | 2019 | — | Número efectivo de muestras para reponderar clases desbalanceadas. |
| Compositional Abilities Emerge Multiplicatively | [2310.09336](https://arxiv.org/abs/2310.09336) | 2023 | — | Fija expectativas para combinaciones de tipos no vistas (heurística, medida en difusión). |
| ⭐ Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis (MSGAN) | [1903.05628](https://arxiv.org/abs/1903.05628) | 2019 | — | Razón de diversidad que registramos (`div`) y pérdida de respaldo contra el colapso. |

## 4. Entrenamiento con pocos datos

| Título | arXiv | Año | Archivo local | Por qué importa |
|---|---|---|---|---|
| ⭐ Differentiable Augmentation for Data-Efficient GAN Training (DiffAugment) | [2006.10738](https://arxiv.org/abs/2006.10738) | 2020 | `papers/limited_data/2006.10738_DiffAugment.pdf` | Augmentar reales y falsos de forma diferenciable; *translation + cutout* para BigGAN condicional (§7). |
| Training Generative Adversarial Networks with Limited Data (StyleGAN2-ADA) | [2006.06676](https://arxiv.org/abs/2006.06676) | 2020 | — | Diagnóstico $r_t$, augmentación no filtrante y la base más fuerte con pocos datos. |
| ⭐ Regularizing Generative Adversarial Networks under Limited Data (LeCam) | [2104.03310](https://arxiv.org/abs/2104.03310) | 2021 | `papers/limited_data/2104.03310_LeCam.pdf` | Regularizador con anclas EMA y su vínculo con la divergencia de LeCam. |
| Improved Consistency Regularization for GANs (bCR/zCR) | [2002.04724](https://arxiv.org/abs/2002.04724) | 2020 | `papers/limited_data/2002.04724_ImprovedCR_bCR_zCR.pdf` | Regularización de consistencia; zCR empuja diversidad por $z$. |
| Consistency Regularization for Generative Adversarial Networks (CR-GAN) | [1910.12027](https://arxiv.org/abs/1910.12027) | 2019 | — | Augmentar solo reales filtra artefactos a las muestras. |
| Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data (APA) | [2111.06849](https://arxiv.org/abs/2111.06849) | 2021 | — | Respaldo barato si D sobreajusta. |
| Data-Efficient Instance Generation from Instance Discrimination (InsGen) | [2106.04566](https://arxiv.org/abs/2106.04566) | 2021 | — | Descartado: trata sprites de la misma especie como negativos. |
| Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis (FastGAN) | [2101.04775](https://arxiv.org/abs/2101.04775) | 2021 | `papers/limited_data/2101.04775_FastGAN.pdf` | Incluye un dataset de Pokémon; su D usa LPIPS preentrenado, lo que choca con «desde cero». |

## 5. Dos vistas, pixel art y trabajos previos con Pokémon

| Título | arXiv | Año | Archivo local | Por qué importa |
|---|---|---|---|---|
| ⭐ Coupled Generative Adversarial Networks (CoGAN) | [1606.07536](https://arxiv.org/abs/1606.07536) | 2016 | `papers/two_view/1606.07536_CoGAN.pdf` | Compartir las primeras capas de G induce correspondencia entre vistas: nuestro tronco común. |
| ⭐ Generative Adversarial Text to Image Synthesis (GAN-CLS) | [1605.05396](https://arxiv.org/abs/1605.05396) | 2016 | `papers/two_view/1605.05396_GAN-INT-CLS_Reed.pdf` | Discriminador *matching-aware*: origen de los negativos desajustados frente/espalda. |
| DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis | [2008.05865](https://arxiv.org/abs/2008.05865) | 2020 | — | Hinge con peso ½ para falsos y desajustados, construidos rotando el batch. |
| Generating Pixel Art Character Sprites using GANs | [2208.06413](https://arxiv.org/abs/2208.06413) | 2022 | `papers/two_view/2208.06413_PixelArtSprites_Pix2Pix_Coutinho.pdf` | Vistas de sprites en pixel art; el canal alfa evita píxeles sueltos fuera de la silueta. |
| A Missing Data Imputation GAN for Character Sprite Generation | [2409.10721](https://arxiv.org/abs/2409.10721) | 2024 | `papers/two_view/2409.10721_SpriteImputation_CollaGAN_Coutinho.pdf` | Deriva de paleta en salidas GAN y cuantización a paleta como post-proceso. |
| Generating Gameplay-Relevant Art Assets with Transfer Learning | [2010.01681](https://arxiv.org/abs/2010.01681) | 2020 | `papers/two_view/2010.01681_PokemonTypes_CVAE_Gonzalez.pdf` | Único trabajo previo con sprites de Pokémon condicionados por tipo (CVAE, 32×32). |
| Autoencoding Generative Adversarial Networks (AEGAN) | [2004.05472](https://arxiv.org/abs/2004.05472) | 2020 | — | Su autor publicó un PokeGAN 96×96 con una cabeza de paleta por muestra. |
| Pixel VQ-VAEs for Improved Pixel Art Representation | [2203.12130](https://arxiv.org/abs/2203.12130) | 2022 | — | Representaciones específicas de pixel art entrenadas con sprites de Pokémon. |
| HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms | [2011.11731](https://arxiv.org/abs/2011.11731) | 2020 | — | Histogramas de color diferenciables (posible pérdida de coherencia de paleta). |
| CollaGAN: Collaborative GAN for Missing Image Data Imputation | [1901.09764](https://arxiv.org/abs/1901.09764) | 2019 | — | Arquitectura multidominio en la que se basa el trabajo de sprites multipose. |
| JointGAN: Multi-Domain Joint Distribution Learning with GANs | [1806.02978](https://arxiv.org/abs/1806.02978) | 2018 | — | Alternativa teórica para aprender distribuciones conjuntas multidominio. |
| Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model | [2310.15110](https://arxiv.org/abs/2310.15110) | 2023 | — | Vistas en mosaico dentro de una imagen para modelar la conjunta (contexto). |
| MVDream: Multi-view Diffusion for 3D Generation | [2308.16512](https://arxiv.org/abs/2308.16512) | 2023 | — | Atención entre vistas para consistencia multivista (contexto). |
| Representation Learning with Contrastive Predictive Coding (InfoNCE) | [1807.03748](https://arxiv.org/abs/1807.03748) | 2018 | — | Cota InfoNCE de la información mutua entre vistas (métrica de coherencia alternativa). |
| On Variational Bounds of Mutual Information | [1905.06922](https://arxiv.org/abs/1905.06922) | 2019 | — | Justifica las cotas de información mutua basadas en un crítico. |

## 6. Evaluación

| Título | arXiv | Año | Archivo local | Por qué importa |
|---|---|---|---|---|
| ⭐ Demystifying MMD GANs (KID) | [1801.01401](https://arxiv.org/abs/1801.01401) | 2018 | `papers/evaluation/1801.01401_KID_Binkowski_MMD-GAN.pdf` | KID = MMD² insesgada con kernel cúbico; el FID está sesgado. Métrica titular (§10.2). |
| Effectively Unbiased FID and Inception Score and where to find them | [1911.07023](https://arxiv.org/abs/1911.07023) | 2019 | — | $\mathbb E[\mathrm{FID}_N]=\mathrm{FID}_\infty+K/N$ con $K$ dependiente del modelo. |
| On Aliased Resizing and Surprising Subtleties in GAN Evaluation (clean-fid) | [2104.11222](https://arxiv.org/abs/2104.11222) | 2021 | `papers/evaluation/2104.11222_cleanfid_Parmar_aliased-resizing.pdf` | Cuantizar y reescalar igual reales y generadas. |
| Reliable Fidelity and Diversity Metrics for Generative Models (densidad/cobertura) | [2002.09797](https://arxiv.org/abs/2002.09797) | 2020 | `papers/evaluation/2002.09797_density-coverage_Naeem.pdf` | Separa fidelidad de diversidad; advierte sobre embeddings de ImageNet fuera de dominio. |
| Improved Precision and Recall Metric for Assessing Generative Models | [1904.06991](https://arxiv.org/abs/1904.06991) | 2019 | — | Precisión/recall con variedades k-NN. |
| Classification Accuracy Score for Conditional Generative Models (CAS) | [1905.10887](https://arxiv.org/abs/1905.10887) | 2019 | `papers/evaluation/1905.10887_CAS_Ravuri-Vinyals.pdf` | Entrenar con generadas y probar con reales; FID no predice el condicionamiento. |
| ⭐ How good is my GAN? (GAN-train / GAN-test) | [1807.09499](https://arxiv.org/abs/1807.09499) | 2018 | — | Nuestro protocolo del clasificador de tipos es *GAN-test*. |
| On the Evaluation of Conditional GANs (FJD) | [1907.08175](https://arxiv.org/abs/1907.08175) | 2019 | `papers/evaluation/1907.08175_conditional-GAN-eval_FJD_DeVries.pdf` | Distancia de Fréchet sobre la conjunta (imagen, condición); idea de métrica de pares. |
| Evaluation Metrics for Conditional Image Generation (WCFID/BCFID) | [2004.12361](https://arxiv.org/abs/2004.12361) | 2020 | — | Descompone el FID en intra- y entre-clases; lo intra-clase detecta colapso condicional. |
| The Role of ImageNet Classes in Fréchet Inception Distance | [2203.06026](https://arxiv.org/abs/2203.06026) | 2022 | — | El FID está atado a las clases de ImageNet: cautela con pixel art. |
| Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models | [2306.04675](https://arxiv.org/abs/2306.04675) | 2023 | — | Recomienda características DINOv2; ninguna métrica separa memorización de subajuste. |
| ⭐ A Non-Parametric Test to Detect Data-Copying in Generative Models | [2004.05675](https://arxiv.org/abs/2004.05675) | 2020 | — | Prueba de copia con conjunto reservado: base de nuestra medida de novedad. |
| On Training Sample Memorization: Lessons from Benchmarking Generative Modeling with a Large-scale Competition (MiFID) | [2106.03062](https://arxiv.org/abs/2106.03062) | 2021 | — | FID penalizado por memorización. |
| The Vendi Score: A Diversity Evaluation Metric for Machine Learning | [2210.02410](https://arxiv.org/abs/2210.02410) | 2022 | — | «Número efectivo» de criaturas distintas por condición. |
| The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS) | [1801.03924](https://arxiv.org/abs/1801.03924) | 2018 | — | Distancia perceptual para la diversidad intra-condición (solo evaluación). |
| DINOv2: Learning Robust Visual Features without Supervision | [2304.07193](https://arxiv.org/abs/2304.07193) | 2023 | — | Extractor alternativo a Inception para métricas y vecinos más cercanos. |
| DINOv3 | [2508.10104](https://arxiv.org/abs/2508.10104) | 2025 | — | Alternativa más reciente a DINOv2 (solo evaluación). |
| Learning Transferable Visual Models From Natural Language Supervision (CLIP) | [2103.00020](https://arxiv.org/abs/2103.00020) | 2021 | — | Puntuación de tipo *zero-shot* como chequeo secundario (no usado en el modelo). |
| Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD) | [2401.09603](https://arxiv.org/abs/2401.09603) | 2024 | — | MMD insesgada sobre embeddings CLIP (alternativa). |

## 7. Descartados por la regla «desde cero»

Ponen redes **preentrenadas dentro del discriminador**. El modelo principal se entrena sin ellas; la variante `vision`
adopta la idea de Vision-aided GAN como discriminador auxiliar.

| Título | arXiv | Año | Archivo local | Por qué importa |
|---|---|---|---|---|
| Projected GANs Converge Faster | [2111.01007](https://arxiv.org/abs/2111.01007) | 2021 | — | D sobre características preentrenadas (EfficientNet); descartado para el modelo principal. |
| StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets | [2202.00273](https://arxiv.org/abs/2202.00273) | 2022 | — | Mismo paradigma de D proyectado + guía por clasificador preentrenado; descartado. |
| Ensembling Off-the-shelf Models for GAN Training (Vision-aided GAN) | [2112.09130](https://arxiv.org/abs/2112.09130) | 2021 | — | Redes de visión preentrenadas como discriminadores; base de la variante `vision`. |

## Repositorios de referencia (solo lectura)

No se copió código de ellos; sirven para contrastar fórmulas e hiperparámetros.

- [ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch): embedding compartido, z jerárquico, EMA y *standing stats*.
- [POSTECH-CVLab/PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN): configuraciones probadas (BigGAN + DiffAugment + LeCam).
- [mit-han-lab/data-efficient-gans](https://github.com/mit-han-lab/data-efficient-gans): DiffAugment oficial.
- [google/lecam-gan](https://github.com/google/lecam-gan): regularizador LeCam oficial.
- [toshas/torch-fidelity](https://github.com/toshas/torch-fidelity): Inception para KID/FID (solo evaluación).

## Cómo volver a descargar los PDF

Cualquier paper se descarga con `https://arxiv.org/pdf/<id>`. Para reconstruir la carpeta `papers/` con las mismas
rutas de la tabla, desde la raíz del repositorio en **Git Bash**:

```bash
while read -r id ruta; do
  mkdir -p "$(dirname "$ruta")"
  curl -sSL -o "$ruta" "https://arxiv.org/pdf/$id"
done <<'EOF'
1606.00709 papers/loss_math/1606.00709_f-gan.pdf
1801.04406 papers/loss_math/1801.04406_mescheder-which-gans-converge.pdf
1807.00734 papers/loss_math/1807.00734_relativistic-gan.pdf
2501.05441 papers/loss_math/2501.05441_r3gan-modern-baseline.pdf
1802.05957 papers/loss_math/1802.05957_spectral-normalization.pdf
1809.11096 papers/architecture/1809.11096_BigGAN.pdf
1805.08318 papers/architecture/1805.08318_SAGAN.pdf
1706.08500 papers/evaluation/1706.08500_FID_Heusel_TTUR.pdf
1802.05637 papers/conditioning/1802.05637_projection-discriminator.pdf
2112.05130 papers/conditioning/2112.05130_poe-gan-multimodal-projection.pdf
2111.01118 papers/conditioning/2111.01118_reacgan.pdf
2206.09479 papers/conditioning/2206.09479_studiogan-benchmark.pdf
2201.06578 papers/conditioning/2201.06578_collapse-by-conditioning.pdf
2006.10738 papers/limited_data/2006.10738_DiffAugment.pdf
2104.03310 papers/limited_data/2104.03310_LeCam.pdf
2002.04724 papers/limited_data/2002.04724_ImprovedCR_bCR_zCR.pdf
2101.04775 papers/limited_data/2101.04775_FastGAN.pdf
1606.07536 papers/two_view/1606.07536_CoGAN.pdf
1605.05396 papers/two_view/1605.05396_GAN-INT-CLS_Reed.pdf
2208.06413 papers/two_view/2208.06413_PixelArtSprites_Pix2Pix_Coutinho.pdf
2409.10721 papers/two_view/2409.10721_SpriteImputation_CollaGAN_Coutinho.pdf
2010.01681 papers/two_view/2010.01681_PokemonTypes_CVAE_Gonzalez.pdf
1801.01401 papers/evaluation/1801.01401_KID_Binkowski_MMD-GAN.pdf
2104.11222 papers/evaluation/2104.11222_cleanfid_Parmar_aliased-resizing.pdf
2002.09797 papers/evaluation/2002.09797_density-coverage_Naeem.pdf
1905.10887 papers/evaluation/1905.10887_CAS_Ravuri-Vinyals.pdf
1907.08175 papers/evaluation/1907.08175_conditional-GAN-eval_FJD_DeVries.pdf
EOF
```

En **PowerShell**, un archivo a la vez:

```powershell
New-Item -ItemType Directory -Force papers\conditioning | Out-Null
Invoke-WebRequest https://arxiv.org/pdf/1802.05637 -OutFile papers\conditioning\1802.05637_projection-discriminator.pdf
```

La copia local tiene además algunos duplicados del mismo paper en otras carpetas temáticas (por ejemplo,
BigGAN también en `papers/conditioning/`); no hacen falta.
