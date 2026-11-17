# La matemática de PokeGAN

> Documento de estudio del modelo. Cuenta **la historia de PokeGAN**: qué problema resolvemos, qué pérdida minimizamos, por qué cada pieza
> está ahí y dónde vive en el código. Cada sección cierra con **«En nuestro código:»**.
>
> Las citas tienen la forma [Autor año, arXiv:ID]. La tabla completa, con enlaces, está en [`papers.md`](papers.md).
> Los resultados marcados como **(derivación propia)** no están en los papers tal cual: son extensiones
> que hicimos y que conviene saber defender.

## Índice

0. [Problema y notación](#0-problema-y-notación)
1. [El juego minimax: D óptimo, JSD y gradientes que se desvanecen](#1-el-juego-minimax-d-óptimo-jsd-y-gradientes-que-se-desvanecen)
2. [Pérdida hinge: qué minimiza y por qué la usamos](#2-pérdida-hinge-qué-minimiza-y-por-qué-la-usamos)
3. [Normalización espectral](#3-normalización-espectral)
4. [GAN condicional y discriminador de proyección (multi-hot por suma)](#4-gan-condicional-y-discriminador-de-proyección-multi-hot-por-suma)
5. [Generador: BatchNorm condicional, z jerárquico y embedding compartido](#5-generador-batchnorm-condicional-z-jerárquico-y-embedding-compartido)
6. [Dos vistas: tronco compartido, composición alfa y negativos desajustados](#6-dos-vistas-tronco-compartido-composición-alfa-y-negativos-desajustados)
7. [Pocos datos: sobreajuste de D, DiffAugment y LeCam](#7-pocos-datos-sobreajuste-de-d-diffaugment-y-lecam)
8. [Colapso por condicionamiento y entrenamiento transicional](#8-colapso-por-condicionamiento-y-entrenamiento-transicional)
9. [Optimización: TTUR, Adam con β1 = 0, EMA y standing statistics](#9-optimización-ttur-adam-con-β1--0-ema-y-standing-statistics)
10. [Evaluación](#10-evaluación)
10b. [Muestreo guiado por D: el GAN es secretamente un modelo de energía](#10b-muestreo-guiado-por-d-el-gan-es-secretamente-un-modelo-de-energía)
10c. [Disección del modelo entrenado y afinado de forma (co-área y soporte)](#10c-disección-del-modelo-entrenado-y-afinado-de-forma-co-área-y-soporte)
11. [Preguntas frecuentes](#11-preguntas-frecuentes)

---

## 0. Problema y notación

**Datos.** Cada muestra es una terna $(x_f, x_b, y)\sim p_d$:

- $x_f, x_b\in[-1,1]^{3\times 96\times 96}$ son el sprite de **frente** y de **espalda** de la misma criatura.
  El fondo es blanco y, tras escalar $u8\mapsto u8/127.5-1$, vale exactamente $+1$.
- Apilamos las vistas por canales: $x=[x_f;x_b]\in[-1,1]^{6\times 96\times 96}$.
- $y\in\lbrace 0,1\rbrace^{18}$ es el vector **multi-hot** de tipos, con $\mathbf 1^\top y\in\lbrace 1,2\rbrace$.
  Escribimos también $c\subset\lbrace 1,\dots,18\rbrace$ para el conjunto de tipos activos ($y=\mathbf 1_c$).
- Hay $\lvert\mathcal C\rvert=18+\binom{18}{2}=171$ condiciones posibles. En los datos curados aparecen 162;
  en el conjunto con que entrena la GAN, 160. Las **11 combinaciones nunca vistas** por el modelo son
  Normal/Roca, Normal/Bicho, Normal/Acero, Normal/Hielo, Veneno/Hielo, Tierra/Hada, Roca/Fantasma,
  Bicho/Dragón, Bicho/Siniestro, Fuego/Hada y Dragón/Hada (9 no existen en todo el dataset; Bicho/Siniestro y
  Dragón/Hada solo están en las especies que reservamos para validación).

**Modelo.**

- Generador: $G(z,y)=\big(G_f(z,y),\,G_b(z,y)\big)$ con $z\sim\mathcal N(0,I_{120})$. **Un solo $z$ produce las
  dos vistas**; esa es la exigencia del enunciado y la base de la coherencia.
- $p_g(x\mid y)$ es la ley de $G(z,y)$. Las etiquetas de las muestras falsas son las del batch real,
  permutadas; así $p_g(y)=p_d(y)$ (lo usaremos en la §4).
- Discriminador (o crítico): $C(x,y)\in\mathbb R$ es un **logit**. Cuando lo leemos como probabilidad,
  $D=\sigma(C)$ con $\sigma(t)=1/(1+e^{-t})$.
- Notación auxiliar: $[t]_+=\max(0,t)$; $\mathrm{TV}(p,q)=\tfrac12\int\lvert p-q\rvert$;
  $\mathrm{KL}(p\Vert q)=\int p\log(p/q)$; $\mathrm{JSD}(p\Vert q)=\tfrac12\mathrm{KL}(p\Vert m)+\tfrac12\mathrm{KL}(q\Vert m)$
  con $m=\tfrac12(p+q)$.

**Objetivo.** Queremos

$$
p_g(x_f,x_b\mid y)=p_d(x_f,x_b\mid y)\quad\text{para toda condición } y\text{, incluidas las no vistas,}
$$

lo que traduce los tres requisitos del problema:

| Requisito del enunciado | Lectura matemática |
|---|---|
| Criaturas **nuevas** condicionadas por 1 o 2 tipos | $p_g(x\mid y)\approx p_d(x\mid y)$ sin copiar los datos de entrenamiento |
| Frente y espalda **coherentes** | igualar la distribución **conjunta** de $(x_f,x_b)$, no solo las marginales |
| **Distintas criaturas** para la misma condición | $z\mapsto G(z,y)$ no es casi constante (sin colapso de modo) |

**En nuestro código:** [`data.py`](../src/pokegan/data.py) (`load_curated`, `to_tensor_pairs`) carga los pares como
`uint8 [N,6,96,96]`; [`types.py`](../src/pokegan/types.py) (`encode`, `all_conditions`) define el multi-hot;
[`train.py`](../src/pokegan/train.py) permuta las etiquetas reales para usarlas en los falsos.

---

## 1. El juego minimax: D óptimo, JSD y gradientes que se desvanecen

La GAN original [Goodfellow et al. 2014, arXiv:1406.2661] plantea

$$
\min_G\max_D\;V(D,G)=\mathbb E_{x\sim p_d}\big[\log D(x)\big]+\mathbb E_{z}\big[\log\big(1-D(G(z))\big)\big].
$$

**Discriminador óptimo.** Con $G$ fijo, $V=\int\big[p_d\log D+p_g\log(1-D)\big]dx$ se maximiza punto a punto.
Para $h(t)=a\log t+b\log(1-t)$ (cóncava en $(0,1)$) se tiene $h'(t)=a/t-b/(1-t)=0$, de donde

$$
D^{\star}(x)=\frac{p_d(x)}{p_d(x)+p_g(x)},\qquad C^{\star}(x)=\log\frac{D^{\star}}{1-D^{\star}}=\log\frac{p_d(x)}{p_g(x)}.
$$

**El discriminador óptimo estima el logaritmo del cociente de densidades.** Esta idea es la que usaremos
para derivar el discriminador de proyección (§4).

**Reducción a Jensen–Shannon.** Sustituyendo $D^{\star}$ y escribiendo $m=\tfrac12(p_d+p_g)$:

$$
V(D^{\star},G)=\mathbb E_{p_d}\Big[\log\frac{p_d}{2m}\Big]+\mathbb E_{p_g}\Big[\log\frac{p_g}{2m}\Big]
=-\log 4+\mathrm{KL}(p_d\Vert m)+\mathrm{KL}(p_g\Vert m)=2\,\mathrm{JSD}(p_d\Vert p_g)-\log 4 .
$$

Como $\mathrm{JSD}\ge 0$ con igualdad solo si $p_g=p_d$, el óptimo global del generador es $p_g=p_d$, donde
$D^{\star}\equiv\tfrac12$. *Matiz:* esto es convexo en el espacio de densidades, no en el de parámetros; por eso la
dinámica (§9) importa tanto como el óptimo.

**Gradiente que se desvanece.** Con $D=\sigma(C)$, la pérdida «saturante» del generador es $\log(1-\sigma(C))$ y

$$
\frac{\partial}{\partial C}\log\big(1-\sigma(C)\big)=-\sigma(C)\approx 0\quad\text{cuando } D \text{ rechaza los falsos con confianza.}
$$

Es justo lo que pasa al inicio. Además, si $p_d$ y $p_g$ viven en variedades de baja dimensión (imágenes; y en
pixel art con ~11 colores por sprite, más aún) existe un discriminador perfecto, $\mathrm{JSD}=\log 2$ es
constante y el gradiente para $G$ tiende a cero [Arjovsky y Bottou 2017, arXiv:1701.04862].

**Pérdida no saturante.** Se reemplaza por $\min_G -\mathbb E_z\log\sigma(C(G(z)))$, cuya derivada respecto al logit
es $-(1-\sigma(C))\approx -1$: mismos puntos fijos, gradiente útil. Pero tiene un costo oculto (Teorema 2.5 de
Arjovsky y Bottou): con $D^{\star}$ fijo,

$$
\mathbb E_z\big[-\nabla_\theta\log D^{\star}(G_\theta(z))\big]=\nabla_\theta\Big[\mathrm{KL}(p_g\Vert p_d)-2\,\mathrm{JSD}(p_g\Vert p_d)\Big].
$$

La KL **inversa** castiga mucho producir muestras irreales y poco olvidar modos (*mode seeking*), y el término
$-2\,\mathrm{JSD}$ tiene el signo «equivocado». Resultado típico: muestras nítidas pero poco diversas, que es
exactamente el riesgo de «misma condición, mismo Pokémon».

**En nuestro código:** no usamos la pérdida logística; esta sección es la base conceptual de la hinge (§2) y de la
proyección (§4), que están en [`losses.py`](../src/pokegan/losses.py) y
[`models/discriminator.py`](../src/pokegan/models/discriminator.py).

---

## 2. Pérdida hinge: qué minimiza y por qué la usamos

$$
\mathcal L_D=\mathbb E_{p_d}\big[1-C(x)\big]_+ +\mathbb E_{p_g}\big[1+C(x)\big]_+,\qquad
\mathcal L_G=-\mathbb E_z\big[C(G(z))\big].
$$

### 2.1 Sin restricciones: variación total

Punto a punto, $D$ minimiza $\ell(t)=p_d[1-t]_+ + p_g[1+t]_+$.

- Fuera de $[-1,1]$ no conviene estar: si $t > 1$, el término real ya vale 0 y el falso sigue creciendo; si
  $t < -1$, al revés. (**Lema de recorte**: $\tilde C=\mathrm{clip}(C,-1,1)$ nunca empeora la pérdida.)
- En $t\in[-1,1]$: $\ell(t)=(p_d+p_g)+t\,(p_g-p_d)$, lineal en $t$, así que $t^{\star}=\operatorname{sign}(p_d-p_g)$ y
  $\ell^{\star}=2\min(p_d,p_g)$.

Como $\int\min(p,q)=1-\mathrm{TV}(p,q)$:

$$
\min_D\mathcal L_D=2-2\,\mathrm{TV}(p_d,p_g).
$$

### 2.2 Con D Lipschitz (lo que impone la normalización espectral)

Si restringimos $\lVert C\rVert_L\le K$, el recorte conserva la constante de Lipschitz, y para $\tilde C\in[-1,1]$ la
pérdida es $2-\big(\mathbb E_{p_d}\tilde C-\mathbb E_{p_g}\tilde C\big)$. Por lo tanto **(derivación propia)**

$$
\min_{\lVert C\rVert_L\le K}\mathcal L_D=2-\beta_K(p_d,p_g),\qquad
\beta_K=\sup_{\lVert f\rVert_L\le K,\ \lVert f\rVert_\infty\le 1}\ \mathbb E_{p_d}f-\mathbb E_{p_g}f .
$$

$\beta_K$ es una métrica tipo Dudley (*bounded-Lipschitz*): interpola entre $K\,W_1$ (si $K\cdot\mathrm{diam}\le 1$, la cota
$\lvert f\rvert\le 1$ no se activa) y $2\,\mathrm{TV}$ (si $K\to\infty$). Por el teorema de la envolvente (Danskin),
como $\mathbb E_{p_d}C^{\star}$ no depende de $\theta$,

$$
\nabla_\theta\mathcal L_G=-\nabla_\theta\,\mathbb E_{p_\theta}\big[C^{\star}\big]=\nabla_\theta\,\beta_K(p_d,p_\theta).
$$

**Conclusión: hinge + SN se comporta como minimizar una métrica débil**, que da gradientes útiles aun con
soportes disjuntos (a diferencia de la JSD). Interpretación geométrica [Lim y Ye 2017, arXiv:1705.02894]: $D$ es
un SVM de margen 1 en el espacio de características y $G$ mueve los falsos en la dirección normal al hiperplano.

### 2.3 Por qué la elegimos

1. **Métrica débil** (arriba): gradientes informativos aunque reales y falsos no se solapen.
2. **Saturación del margen:** una vez que un real tiene $C > 1$ (o un falso $C < -1$), $D$ deja de empujarlo. Eso
   limita la sobreconfianza de $D$, que es el síntoma principal del sobreajuste con pocos datos (§7).
3. **Receta probada** en exactamente nuestro régimen: SNGAN, BigGAN, SAGAN y las recetas de pocos datos de
   DiffAugment y LeCam usan hinge + SN + proyección [Miyato et al. 2018, arXiv:1802.05957; Brock et al. 2018, arXiv:1809.11096].
4. **Barata y con pocos hiperparámetros:** no requiere doble retropropagación (como R1 o WGAN-GP), y SN pide menos
   ajuste que la penalización de gradiente [Kurach et al. 2018, arXiv:1807.04720].

*Alternativa descartada hoy:* RpGAN + R1 + R2 (R3GAN) tiene mejor teoría (convergencia local, sin «cuencas malas»)
[Huang et al. 2025, arXiv:2501.05441], pero es incompatible con BatchNorm, cuesta ~2× por paso de $D$ y sus
recetas recorren decenas de millones de imágenes. Queda como trabajo futuro (§9 explica la dinámica).

**En nuestro código:** [`losses.py`](../src/pokegan/losses.py): `d_hinge(real, fake, mis=None, mis_weight=0.5)` y
`g_hinge(fake)`. Los tests fijan los signos: `d_hinge(+2, -2) == 0`, `d_hinge(0, 0) == 2`, `g_hinge(-1) == 1`.

---

## 3. Normalización espectral

La norma espectral de una matriz es su mayor valor singular,
$\sigma(W)=\max_{h\ne 0}\lVert Wh\rVert_2/\lVert h\rVert_2$, y es la constante de Lipschitz de $h\mapsto Wh$. Para
una red $f=W_{L+1}\circ a_L\circ W_L\circ\cdots\circ a_1\circ W_1$ con activaciones 1-Lipschitz (ReLU), como
$\lVert g_1\circ g_2\rVert_L\le\lVert g_1\rVert_L\,\lVert g_2\rVert_L$,

$$
\lVert f\rVert_L\le\prod_{l=1}^{L+1}\sigma(W_l).
$$

La normalización espectral reemplaza cada peso por $\bar W=W/\sigma(W)$, de modo que la cota queda en 1
[Miyato et al. 2018, arXiv:1802.05957].

**Iteración de potencia.** Calcular la SVD en cada paso es caro. Con $W=\sum_i\sigma_i u_iv_i^\top$ se itera

$$
\tilde v\leftarrow\frac{W^\top\tilde u}{\lVert W^\top\tilde u\rVert},\qquad
\tilde u\leftarrow\frac{W\tilde v}{\lVert W\tilde v\rVert},\qquad
\hat\sigma=\tilde u^\top W\tilde v .
$$

Como $(W^\top W)^k v_0=\sum_i\sigma_i^{2k}\langle v_0,v_i\rangle v_i$, el error decae como $(\sigma_2/\sigma_1)^{2k}$. Y
como $W$ cambia poco entre pasos, $\tilde u$ se **reutiliza**: basta **una iteración por paso**.

**Qué hace en el gradiente.** Con $\delta$ el gradiente que llega a la salida de la capa,

$$
\frac{\partial V}{\partial W}=\frac{1}{\sigma(W)}\Big(\hat{\mathbb E}\big[\delta h^\top\big]-\lambda\,u_1v_1^\top\Big),\qquad
\lambda=\hat{\mathbb E}\big[\delta^\top\bar W h\big]:
$$

el segundo término es una penalización adaptativa sobre la dirección singular dominante, que impide que la capa se
vuelva hipersensible en una sola dirección.

**Matices honestos (derivación propia).**

- *Convoluciones:* PyTorch normaliza el kernel remodelado $W_r\in\mathbb R^{C_{out}\times C_{in}k^2}$. Como cada píxel
  aparece en a lo sumo $k^2$ parches de `im2col`, $\lVert\mathrm{Conv}\rVert_{op}\le k\,\sigma(W_r)$: la cota no es
  exactamente 1 (el valor exacto está en [Sedghi et al. 2018, arXiv:1805.10408]).
- *Bloques residuales:* $\mathrm{Lip}(x+F(x))\le 1+\mathrm{Lip}(F)$; la suma global y la autoatención también
  agrandan la cota.

Por eso hay que presentarla como **un regularizador Lipschitz fuerte**, no como una restricción exacta. Lo que
sí garantiza es que los logits no puedan crecer sin control con pequeños cambios de la entrada, y eso hace que la
hinge se comporte como la métrica de la §2.2. La usamos también en $G$, como SAGAN y BigGAN, porque estabiliza el
condicionamiento [Zhang et al. 2018, arXiv:1805.08318].

**Precisión numérica.** El $\varepsilon$ por defecto de SN en PyTorch ($10^{-12}$) se redondea a 0 en fp16, y bajo
`autocast` la iteración de potencia se calcula en media precisión. Entrenamos en **fp32 + TF32**.

**En nuestro código:** [`models/layers.py`](../src/pokegan/models/layers.py): `snconv` y `snlinear` envuelven
`torch.nn.utils.parametrizations.spectral_norm` (eps 1e-6) sobre pesos con inicialización ortogonal. Todas las capas
lineales y convolucionales de $D$ (incluidas $\psi$ y la proyección $V$) y de $G$ (incluidas las que predicen las
ganancias de la CBN) usan SN; la única excepción es el embedding de tipos $E$ de $G$, que se normaliza con RMSNorm.

---

## 4. GAN condicional y discriminador de proyección (multi-hot por suma)

### 4.1 Qué estima un discriminador condicional

Con etiquetas, $D$ ve pares $(x,y)$. Repitiendo la §1 sobre la conjunta, el logit óptimo es

$$
C^{\star}(x,y)=\log\frac{p_d(x,y)}{p_g(x,y)}
=\underbrace{\log\frac{p_d(y\mid x)}{p_g(y\mid x)}}_{r(y\mid x)}+\underbrace{\log\frac{p_d(x)}{p_g(x)}}_{r(x)} .
$$

Como los falsos llevan las etiquetas del batch real, $p_g(y)=p_d(y)$ y $C^{\star}=\log\frac{p_d(x\mid y)}{p_g(x\mid y)}$:
**D compara exactamente las distribuciones condicionales que queremos igualar.**

### 4.2 El caso de una etiqueta: proyección

Miyato y Koyama [2018, arXiv:1802.05637] suponen que $p(y\mid x)$ es un softmax log-lineal sobre unas
características $\phi(x)$: $\log p(y=k\mid x)=v_k^\top\phi(x)-\log Z(x)$. Entonces

$$
r(y=k\mid x)=(v^{d}_k-v^{g}_k)^\top\phi(x)-\log\frac{Z_d(x)}{Z_g(x)} .
$$

El último término depende **solo de $x$** y se absorbe, junto con $r(x)$, en una función $\psi(\phi(x))$:

$$
C(x,y)=\psi\big(\phi(x)\big)+y^\top V\phi(x)\qquad(y\ \text{one-hot},\ V=V_d-V_g).
$$

### 4.3 Nuestra extensión: conjuntos de tipos ⇒ **suma** (derivación propia)

Modelamos la probabilidad de un **conjunto** de tipos $c$ como log-lineal sobre los 171 conjuntos admisibles:

$$
p(c\mid x)=\frac{\exp\big(\sum_{t\in c}a_t(x)\big)}{Z(x)},\qquad Z(x)=\sum_{c'\in\mathcal C}\exp\Big(\sum_{t\in c'}a_t(x)\Big).
$$

El log-cociente es $r(c\mid x)=\sum_{t\in c}\big[a^d_t(x)-a^g_t(x)\big]-\log\frac{Z_d(x)}{Z_g(x)}$. Tomando
$a^d_t-a^g_t=v_t^\top\phi(x)$ y absorbiendo lo que solo depende de $x$:

$$
\boxed{\;C(x,y)=\psi\big(\phi(x)\big)+\sum_{t\in c}v_t^\top\phi(x)=\psi\big(\phi(x)\big)+y^\top V\phi(x)\;}
\qquad\text{con } y \text{ multi-hot } \textbf{sin normalizar}.
$$

- **No hace falta suponer independencia entre tipos:** la restricción $\lvert c\rvert\le 2$ vive en $Z(x)$, que solo
  depende de $x$. (Con Bernoullis independientes por tipo se llega a la misma suma; PoE-GAN obtiene la misma
  forma para varias condiciones [Huang et al. 2021, arXiv:2112.05130].)
- **Producto de expertos:** $\frac{p_d(x\mid c)}{p_g(x\mid c)}\propto e^{\psi}\prod_{t\in c}e^{v_t^\top\phi(x)}$. Es una
  **conjunción**: «parece Fuego **y** parece Volador». Las energías se suman [Du et al. 2020, arXiv:2004.06030].
- **Monotipo vs. dual:** cualquier término que dependa de la cardinalidad, $g(\lvert c\rvert,x)$ con
  $\lvert c\rvert\in\lbrace 1,2\rbrace$, es lineal en $\mathbf 1^\top y$ y se absorbe en $V$; no hace falta una bandera «mono/dual».

### 4.4 ¿Por qué no la media? (derivación propia)

Supongamos un $D$ que promedia, $\psi+\frac{1}{\lvert c\rvert}\sum_{t\in c}{v'_t}^{\top}\phi$, permitiéndole además un término de
cardinalidad $\lvert c\rvert\,u^\top\phi$. Para que reproduzca el logit óptimo $\sum_{t\in c}v_t^\top\phi$:

- monotipos: $v_t'+u=v_t$, o sea $v_t'=v_t-u$;
- duales: $\tfrac12(v_a'+v_b')+2u=v_a+v_b\ \Rightarrow\ \tfrac12(v_a+v_b)+u=v_a+v_b\ \Rightarrow\ v_a+v_b=2u$ para todo $a\ne b$.

Con $18\ge 3$ tipos, $v_a+v_b=v_a+v_c$ implica $v_b=v_c$: **todas las $v_t$ son iguales**, es decir, el término
condicional ya no distingue tipos. La media solo representa el caso degenerado: **en D va la suma**. (En $G$ da
igual, ver §5.)

### 4.5 ¿Por qué no una tabla de 171 combinaciones?

- **Datos por parámetro.** Cada tipo aparece en 419–2254 pares y 60–161 especies; en cambio la combinación
  mediana tiene 20 pares y 4 especies, y 18 combinaciones tienen **una sola especie** (dataset completo). Un
  embedding por combinación memorizaría esa especie (Zapdos es el 66 % de Eléctrico/Volador).
- **Combinaciones no vistas.** Una tabla no tiene vector entrenado para las 11 combinaciones que el modelo nunca vio.
  Con la suma, $C(x,\mathbf 1_a+\mathbf 1_b)=\psi+v_a^\top\phi+v_b^\top\phi$ **está definido** y usa dos expertos
  bien entrenados: es generalización composicional.
- **Expectativa honesta.** La calidad en combinaciones raras o no vistas suele ser peor: la habilidad composicional
  emerge aproximadamente de forma multiplicativa [Okawa et al. 2023, arXiv:2310.09336] (heurística, medida en
  difusión sobre datos sintéticos). Por eso las evaluamos aparte.
- Un MLP profundo sobre $y$ dentro de $D$ (como el de StyleGAN2-ADA) rompería la aditividad.

### 4.6 ¿Por qué no AC-GAN?

AC-GAN añade a $D$ un clasificador $Q^c(y\mid x)$ y le pide a $G$ que sus muestras se clasifiquen bien
[Odena et al. 2016, arXiv:1610.09585]. Sobre los falsos, ese término se descompone así
[Gong et al. 2019, arXiv:1907.02690]:

$$
\mathbb E_{Q}\big[-\log Q^c(Y\mid X)\big]=H_Q(Y\mid X)+\mathrm{KL}\big(Q_{Y\mid X}\,\Vert\,Q^c_{Y\mid X}\big)
$$

(basta sumar y restar $\log Q(Y\mid X)$). Al minimizarlo, **$G$ minimiza la entropía condicional**
$H_Q(Y\mid X)$: empuja cada muestra al interior de la región de decisión de su clase (Teorema 1 de TAC-GAN:
$Q(Y\mid x)$ se vuelve determinista). El resultado son criaturas estereotipadas y poca diversidad intra-tipo,
justo lo contrario del tercer requisito; y en Pokémon los tipos se solapan visualmente (aves Normal/Volador vs.
Agua/Volador, Agua vs. Hielo). Evidencia: en CUB200 (pocos datos) AC-GAN obtiene FID 60.7 frente a 18.3 de la
proyección [Kang et al. 2021, arXiv:2111.01118]; Miyato y Koyama reportan que AC-GAN con 1000 clases generaba «una
sola imagen para la mayoría de las clases». La proyección no tiene ese término de entropía.

**En nuestro código:** [`models/discriminator.py`](../src/pokegan/models/discriminator.py): `Discriminator.forward(x, y)`
devuelve `(uc, proj)` con `uc = ψ(φ)` (`snlinear(512→1)`) y `proj = ⟨V y, φ⟩` (`snlinear(18→512, bias=False)` sobre el
multi-hot crudo); $C_c=$ `uc + proj`. Un test comprueba la linealidad `proj(y_a + y_b) == proj(y_a) + proj(y_b)`.

---

## 5. Generador: BatchNorm condicional, z jerárquico y embedding compartido

**Embedding compartido de tipos.** $e(y)=E^\top y=\sum_{t\in c}e_t$ con $E\in\mathbb R^{18\times 128}$ sin sesgo, y

$$
\hat e=\lambda_t\cdot\mathrm{RMSNorm}(e),\qquad \mathrm{RMSNorm}(u)=\frac{u}{\sqrt{\tfrac1d\lVert u\rVert^2+\epsilon}} .
$$

RMSNorm es invariante a escalar por $\alpha > 0$, así que en $G$ **suma y media son lo mismo**, y un tipo dual no
llega con «el doble de fuerza» que un monotipo. $\lambda_t$ es el factor de transición de la §8.

**z jerárquico.** $z=[z_0;z_1;\dots;z_5]$ con $z_\ell\in\mathbb R^{20}$. $z_0$ alimenta la capa lineal inicial
($\to 512\times3\times3$) y cada bloque recibe su propio trozo. Así cada escala (silueta gruesa, partes, detalles)
recibe ruido directamente (*skip-z*).

**BatchNorm condicional (CBN).** En cada bloque $\ell$, con $s_\ell=[z_\ell;\hat e]$:

$$
\mathrm{CBN}(h\mid s_\ell)=\big(1+W^\ell_\gamma s_\ell\big)\odot\frac{h-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+W^\ell_\beta s_\ell .
$$

La ganancia está centrada en 1, así que la modulación parte cerca de una BatchNorm sin condición [de Vries et al. 2017,
arXiv:1707.00683; Brock et al. 2018, arXiv:1809.11096]. Interpretación:

- El **tipo modula todas las escalas**, desde la forma hasta la paleta fina (el color es la señal de tipo más
  fuerte que medimos, §7.4).
- Cada **$z_\ell$ controla la variación a su escala**: distintos $z$, distintas criaturas.
- $G$ es no lineal en $\hat e$, así que puede modelar interacciones entre los dos tipos aunque $e$ sea aditivo: la suma
  es un **sesgo inductivo** hacia la composición, no una restricción.
- BigGAN reporta que el embedding compartido acelera el entrenamiento un 37 % y que skip-z suma ~4 % de calidad y
  18 % de velocidad.

**Detalles de arquitectura con justificación.**

- *Sobremuestreo nearest ×2 + conv 3×3* en vez de convolución transpuesta: la transpuesta con $k\bmod s\ne 0$
  solapa de forma desigual y produce tableros de ajedrez; nearest es además el interpolador natural del pixel
  art (bloques constantes).
- *Autoatención a 24×24* [Zhang et al. 2018, arXiv:1805.08318]: $o_j=W_o\sum_i\mathrm{softmax}_i\big(f(h_i)^\top g(h_j)\big)\,v(h_i)$,
  $h'=h+\gamma o$ con $\gamma_0=0$. $24/96=32/128$ es el análogo de la mejor ubicación de SAGAN («feat32»), con
  solo $N=576$ posiciones.
- *Inicialización ortogonal* (BigGAN).
- *BatchNorm acopla el batch:* en entrenamiento normaliza con estadísticas del batch; para muestrear usamos modo
  `eval` con *standing statistics* (§9). Por lo mismo, no se usa acumulación de gradientes.
- *Truncamiento* (opcional al generar): muestrear $z$ de una normal truncada $\lvert z_i\rvert\le\tau$ sube la fidelidad y
  baja la diversidad. Las **métricas se calculan sin truncar**.

**En nuestro código:** [`models/layers.py`](../src/pokegan/models/layers.py): `ConditionalBN`, `GBlock`, `SelfAttention`;
[`models/generator.py`](../src/pokegan/models/generator.py): `Generator(ch=32, z_dim=120, n_chunks=6, emb_dim=128)`,
`forward(z, y, lam)`.

---

## 6. Dos vistas: tronco compartido, composición alfa y negativos desajustados

### 6.1 Por qué D tiene que ver el par

Por la regla de la cadena de la KL (todo condicionado a $y$):

$$
\mathrm{KL}\big(p_{fb}\Vert q_{fb}\big)=\mathrm{KL}\big(p_f\Vert q_f\big)+\mathbb E_{x_f\sim p_f}\Big[\mathrm{KL}\big(p(x_b\mid x_f)\Vert q(x_b\mid x_f)\big)\Big].
$$

Dos discriminadores separados (uno por vista) solo controlan las marginales. Un generador con vistas
**independientes**, $q=p_f\,p_b$, tendría marginales perfectas y, sin embargo,
$I_q(X_f;X_b\mid Y)=0$: frente y espalda de criaturas distintas. **La coherencia es exactamente la dependencia**
entre vistas, y solo un $D$ que vea el par la controla: con él, el óptimo es $2\,\mathrm{JSD}(p_{fb}\Vert q_{fb})-\log 4$,
que se anula solo si la **conjunta** coincide.

### 6.2 Generador: tronco compartido + dos cabezas (idea de CoGAN)

$G_v=H_v\circ T$: un tronco $T$ común hasta $64\times48\times48$ y una cabeza $H_v$ por vista (pesos separados)
que sube a $96\times96$ y emite 4 canales (RGB + alfa). CoGAN muestra que compartir las primeras capas de $G$
induce correspondencia entre dominios **incluso sin pares**, y que el acuerdo crece con el número de capas
compartidas [Liu y Tuzel 2016, arXiv:1606.07536]. Nosotros sí tenemos pares, así que combinamos ese sesgo
inductivo con un $D$ conjunto. Cada cabeza tiene su propia geometría: las vistas **no** están alineadas píxel a
píxel (IoU de siluetas frente/espalda: mediana 0.415).

### 6.3 Composición alfa sobre blanco

Cada cabeza produce $(\mathrm{rgb}_v,a_v)$ y

$$
x_v=\sigma(a_v)\odot\tanh(\mathrm{rgb}_v)+\big(1-\sigma(a_v)\big)\odot\mathbf 1 .
$$

- Es una combinación convexa de $\tanh(\cdot)\in(-1,1)$ y del blanco $+1$, así que $x_v\in(-1,1]$.
- **Fondo blanco exacto:** como $x_v\ge 1-2\sigma(a_v)$, basta $\sigma(a_v)\le 1/510\approx 0.002$ para que, al
  cuantizar a `uint8`, el píxel sea exactamente 255.
- **Por qué importa:** el 82–85 % de los píxeles reales son exactamente blancos; un `tanh` nunca llega a 1, y un
  fondo «casi blanco» sería una pista trivial para $D$.
- **Separa «dónde» de «qué color»:** $\partial x_v/\partial\,\mathrm{rgb}_v=\sigma(a_v)(1-\tanh^2)$, así que en el fondo
  ($\sigma(a)\approx 0$) el color no recibe gradiente; la silueta la decide $a_v$.

### 6.4 Discriminador conjunto de 6 canales

$D$ recibe $[x_f;x_b]$ apilado: compara colores de ambas vistas desde la primera convolución, y hacia $12\times12$
su campo receptivo ya cubre el sprite completo. Es la mitad de caro que un $D$ siamés en las capas de alta
resolución (el siamés queda como trabajo futuro).

### 6.5 Negativos desajustados

Formamos pares reales **incoherentes** $\tilde x_i=[x_f^{(i)};x_b^{(j)}]$ con la etiqueta $y_i$, donde $j$ sale de un
desarreglo del batch (sin puntos fijos) con **otra especie**, y los tratamos como falsos con peso $\tfrac12$
(idea del discriminador *matching-aware* de GAN-CLS [Reed et al. 2016, arXiv:1605.05396] y de DF-GAN
[Tao et al. 2020, arXiv:2008.05865]). Así $D$ aprende explícitamente que «frente de uno + espalda de otro» es falso.

¿Cambia esto lo que aprende $G$? No, si los negativos están fuera del soporte de los datos.

**Proposición (soporte disjunto ⇒ no sesgan el óptimo de G).** Sea $q$ la ley de los negativos desajustados con
$\mathrm{supp}\,q\cap\mathrm{supp}\,p_d=\varnothing$, y sea $w > 0$ su peso. En la versión logística, $D$ maximiza

$$
\mathbb E_{p_d}\log D+\mathbb E_{p_g}\log(1-D)+w\,\mathbb E_{q}\log(1-D).
$$

Entonces el valor $V_w(G)=\max_D(\cdot)$ coincide con el de la GAN sin negativos, $V_w(G)=2\,\mathrm{JSD}(p_d\Vert p_g)-\log 4$,
para **todo** $p_g$; en particular su minimizador sigue siendo $p_g=p_d$.

*Demostración.* Punto a punto, $D^{\star}=p_d/(p_d+p_g+wq)$ y

$$
V_w(G)=\int\Big[p_d\log\frac{p_d}{p_d+p_g+wq}+(p_g+wq)\log\frac{p_g+wq}{p_d+p_g+wq}\Big]dx .
$$

Sea $A=\mathrm{supp}\,p_d$. **Fuera de $A$**, $p_d=0$ y el integrando es $(p_g+wq)\log 1=0$: los negativos no
aportan nada porque $D^{\star}=0$ ahí. **Dentro de $A$**, $q=0$ y el integrando es
$p_d\log\frac{p_d}{p_d+p_g}+p_g\log\frac{p_g}{p_d+p_g}$, idéntico al de la GAN original (que fuera de $A$ también
vale 0). Luego $V_w(G)=V(D^{\star},G)=2\,\mathrm{JSD}(p_d\Vert p_g)-\log 4$. $\blacksquare$

Con hinge vale el mismo argumento: fuera de $A$ basta $t\le -1$ para pérdida 0, y dentro de $A$ no hay negativos,
así que $\min_D\mathcal L_D=2-2\,\mathrm{TV}(p_d,p_g)$ igual que en la §2.1 y el $D^{\star}$ sobre $A$ no cambia.

**Entonces, ¿para qué sirven?** La proposición habla del óptimo con $D$ de capacidad infinita. Con un $D$ real
(Lipschitz, pocos datos), los negativos le dan supervisión directa sobre la coherencia: sus características
$\phi$ aprenden a detectar «paleta o anatomía que no coinciden», y por continuidad eso se convierte en gradiente para
$G$ hacia pares coherentes. Es supervisión gratis (pares reales que ya tenemos).

**Cuándo sí sesgan.** Si el negativo cae **en** el soporte de los datos. Por eso: (i) exigimos especie distinta
(un intercambio al azar coincide en especie solo el 0.27 % de las veces); (ii) **no** usamos negativos de «tipo
equivocado», porque el 14.3 % de las combinaciones al azar comparten un tipo con la verdadera y empujarían a $D$
hacia un comportamiento tipo AC-GAN. (Si los negativos fueran el producto de marginales completo, que sí se
solapa con los datos, el óptimo se sesgaría hacia pares «demasiado coherentes»: esa era la preocupación de la nota 03.)

### 6.x El peso de los negativos importa: un colapso real y su diagnóstico

**Lo que pasó.** La primera corrida usaba $H(C)=\mathbb E_{\text{real}}[1-C]_+ +1\cdot\mathbb E_{\text{falso}}[1+C]_+ +\tfrac12\,\mathbb E_{\text{des}}[1+C]_+$.
Desde el paso ~100, $D$ devolvió la misma salida para todo: $C(\text{real})=C(\text{falso})=C(\text{des})\approx-0.96$,
pérdida $\approx 2$, $r_t=-1$. En el checkpoint del paso 2000 la varianza de las activaciones de $D$ caía capa a capa
(0.32 → 0.18 → 0.024) y el 80 % de las ReLU finales estaban muertas, aunque los falsos eran claramente distintos
(35 % de píxeles blancos frente a 73 % en los reales).

**Por qué (punto estacionario de un $D$ constante).** Si $D\equiv c$, la derivada de la hinge es
$$
\frac{\partial H}{\partial c}=-w_r\,\mathbb 1[c<1]+w_f\,\mathbb 1[c>-1]+w_m\,\mathbb 1[c>-1].
$$
Con $w_r=1$, $w_f=1$, $w_m=\tfrac12$ vale $+\tfrac12>0$ en todo $(-1,1)$: el descenso empuja $c$ hasta $-1$, donde las
bisagras negativas se **apagan** (gradiente 0). Ahí $D$ queda estacionado. Para salir necesitaría rasgos que separen
reales de negativos, pero los desajustados son **reales vista por vista**: todo rasgo de «realismo» que sube $C$ en
los reales también lo sube en los desajustados, y $G$ —que se adapta en cada paso— persigue esos mismos rasgos.
Con pesos balanceados como en GAN-CLS [Reed et al. 2016, arXiv:1605.05396], $w_r=1$, $w_f=w_m=\tfrac12$, la
derivada es $0$ en $(-1,1)$: no hay tirón hacia la zona muerta.

**Cómo se confirmó (ablación, 400 pasos, un factor a la vez).**

| Variante | $C$ real | $C$ falso | $C$ des. |
|---|---|---|---|
| original ($w_f=1$) | −0.92 | −0.94 | −0.94 (colapso) |
| sin TF32 | −0.92 | −0.94 | −0.94 (descartado) |
| $G$ congelado | +0.97 | −5.54 | −0.00 (sano) |
| sin desajustados | +0.19 | −0.03 | +0.18 |
| balanceado + $n_{dis}=2$ | +0.51 | +0.13 | **−0.45** (sano; aprende coherencia) |

**Primera corrección:** $w_f=w_m=\tfrac12$ y $n_{dis}=2$ (dos pasos de $D$ por paso de $G$, como BigGAN). Funcionó
al inicio, pero una revisión matemática independiente mostró un **segundo problema** en el régimen tardío.

**La trampa espejo y la identificabilidad.** Cuando $D$ ya hunde los desajustados bajo $-1$, su bisagra se apaga
y quedan activos real ($w_r=1$) y falso ($w_f=\tfrac12$): $\partial H/\partial c=-1+\tfrac12<0$ y un $D$ constante
deriva hacia $+1$. Peor: con hinge y $D$ no paramétrico, el óptimo punto a punto en el soporte es
$C^\star=\mathrm{sign}(w_r p-w_f q)$, así que **cualquier** $q\le (w_r/w_f)\,p=2p$ deja a $D$ en $+1$ en todo el
soporte: $G$ podría abandonar hasta la mitad de la masa de los datos (modos) sin penalización. (Con la pérdida
logística de Reed esto no pasa; es específico de la hinge.)

**Corrección final: término desajustado por pares (relativista).**
$$
H(C)=\mathbb E_{\text{real}}[1-C]_+ +\mathbb E_{\text{falso}}[1+C]_+ +\tfrac12\,\mathbb E_i\big[1-C(f_i,b_i)+C(f_i,b_j)\big]_+ .
$$
El sesgo de $D$ aparece con signos opuestos dentro del último término y **se cancela**: un $D$ constante tiene
$\partial H/\partial c=-1+1+0=0$ en todo régimen, $w_r=w_f$ devuelve el equilibrio único $q=p$, y el término solo
reacciona a diferencias entre el par coherente y el incoherente con el **mismo frente**: es una señal pura de
coherencia frente/espalda. Se aplicó reanudando desde el paso 6000 de la corrida sana.

Lección: en un GAN, un detalle de pesos que parece inocente puede crear un punto estacionario degenerado; se
detecta mirando la separación de logits ($C$ real − $C$ falso), no solo la pérdida, y se previene escribiendo la
derivada de la pérdida para un $D$ constante en cada régimen (hay un test por cada caso en `tests/test_losses.py`).

**En nuestro código:** [`models/generator.py`](../src/pokegan/models/generator.py) (tronco, dos cabezas y composición alfa;
un test fuerza alfa ≈ 0 y comprueba salida exactamente blanca); [`models/discriminator.py`](../src/pokegan/models/discriminator.py)
(entrada de 6 canales); [`losses.py`](../src/pokegan/losses.py): `derangement_other_species(species)` (desarreglo del
batch más una máscara «especie distinta»; las filas de la misma especie se descartan) y el término `mis` de `d_hinge`;
[`train.py`](../src/pokegan/train.py) arma los desajustados con la mitad del batch real y los pasa por la misma
DiffAugment pareada que a reales y falsos, en ambas cabezas ($C_{uc}$ y $C_c$).

---

## 7. Pocos datos: sobreajuste de D, DiffAugment y LeCam

### 7.1 ¿Cuántos datos tenemos realmente?

Entrenamos con 7 739 pares, pero muchos son casi duplicados (mismo Pokémon en varios juegos, versión shiny).
Contando cada forma como un grupo con $n_k$ pares, el tamaño efectivo de Kish es

$$
N_{\mathrm{eff}}=\frac{\big(\sum_k n_k\big)^2}{\sum_k n_k^2}\approx 815\ \text{(subconjunto curado; 558 con todo el dataset)}.
$$

Es el régimen de 1k–5k imágenes donde ADA documenta que $D$ sobreajusta [Karras et al. 2020, arXiv:2006.06676].

### 7.2 Por qué sobreajusta D y cómo lo vemos

Con la distribución empírica $\hat p_d=\frac1N\sum_i\delta_{x_i}$, el $D$ óptimo vale «real» solo **sobre** los
$x_i$: memoriza y le da a $G$ gradientes que empujan a copiar. Diagnósticos (ADA):

$$
r_t=\mathbb E\big[\operatorname{sign}C(T x_{\text{real}})\big]\ \ (\text{sano} \lesssim 0.6),\qquad
r_v=\frac{\mathbb E[C_{\text{train}}]-\mathbb E[C_{\text{val}}]}{\mathbb E[C_{\text{train}}]-\mathbb E[C_{\text{gen}}]}\ \ (0=\text{sin sobreajuste},\ 1=\text{total}).
$$

$r_v$ necesita especies **nunca vistas**: por eso reservamos el 5 % de las especies. En nuestro registro usamos una
versión más simple: `r_v` es la misma medida que $r_t$, pero sobre 256 pares de especies reservadas. Si `r_t` sube
y `r_v` no, $D$ está memorizando el conjunto de entrenamiento.

### 7.3 DiffAugment: augmentar reales **y** falsos, de forma diferenciable

Con una transformación aleatoria $T_\varphi$ [Zhao et al. 2020, arXiv:2006.10738]:

$$
\mathcal L_D=\mathbb E\big[1-C(T_\varphi x,y)\big]_+ +\mathbb E\big[1+C(T_\varphi G(z,y),y)\big]_+,\qquad
\mathcal L_G=-\mathbb E\big[C(T_\varphi G(z,y),y)\big].
$$

- **Solo reales:** $D$ compara $T_\#p_d$ con $p_g$ y el óptimo de $G$ es $p_g=T_\#p_d$: $G$ aprende los artefactos
  (en BigGAN/CIFAR-10 el FID pasa de 9.59 a 49.38).
- **Reales y falsos, pero sin derivar a través de $T$ en el paso de $G$:** diverge.
- **DiffAugment completo:** el juego compara $T_\#p_g$ con $T_\#p_d$. Si el operador $\mu\mapsto T_\#\mu$ es
  **inyectivo**, $T_\#p_g=T_\#p_d\Rightarrow p_g=p_d$: **no hay fuga** y $G$ no aprende a imitar la augmentación.

### 7.4 ¿Cuándo es inyectivo? Análisis espectral (derivación propia)

Sea un grupo abeliano de transformaciones $\tau$ elegidas con probabilidades $w(\tau)$, y
$\mathcal A_w\mu=\sum_\tau w(\tau)\,\tau_\#\mu$. Para funciones de prueba que se transforman como caracteres,
$\psi_k\circ\tau=\chi_k(\tau)\psi_k$:

$$
\langle\mathcal A_w\mu,\psi_k\rangle=\hat w(k)\,\langle\mu,\psi_k\rangle,\qquad\hat w(k)=\sum_\tau w(\tau)\,\chi_k(\tau).
$$

$\mathcal A_w$ es inyectivo si y solo si $\hat w(k)\ne 0$ para todo $k$, y la señal que recibe $G$ sobre la
componente $k$ se atenúa por $\lvert\hat w(k)\rvert$.

- **Traslación entera uniforme en $\lbrace -s,\dots,s\rbrace$** (modelo circular $\mathbb Z_{96}$):
  $\hat w(k)=\dfrac{\sin\big(\pi k(2s+1)/96\big)}{(2s+1)\sin(\pi k/96)}$, que solo se anula si $96\mid k(2s+1)$ con
  $96\nmid k$. Con $s=12$, $\gcd(25,96)=1$: **no hay fuga**. Además, con relleno **blanco** y un sprite con margen
  $\ge s$, trasladar equivale exactamente a un desplazamiento cíclico (el 95.5 % de los sprites tiene ≥ 8 px de
  margen y el 81 % ≥ 12 px): es casi una simetría exacta de los datos. El relleno gris (0) no lo sería.
- **Volteo horizontal con probabilidad $p$ dentro del pipeline de D (corrección importante).** El grupo es
  $\lbrace e,F\rbrace$ y sus caracteres valen $\chi(F)=\pm 1$. Descomponiendo $\mu$ en parte simétrica
  ($F_\#\mu_s=\mu_s$) y antisimétrica ($F_\#\mu_a=-\mu_a$):

  $$
  \mathcal A_p\mu=(1-p)\,\mu+p\,F_\#\mu=\mu_s+(1-2p)\,\mu_a,\qquad\hat w\in\lbrace 1,\ 1-2p\rbrace .
  $$

  Con $p=0.5$, $\hat w=0$: $D$ ve $\tfrac12(\mu+F_\#\mu)$ y **no puede distinguir una distribución de su espejo**; la
  orientación «se fuga». La fórmula $1-p$ que aparecía en la nota 02 corresponde a otra receta: con probabilidad
  $p$ se sortea un elemento **uniforme de todo el grupo, identidad incluida** (rotación de 90° entre los 4 ángulos,
  giro de matiz uniforme, o un volteo «a cara o cruz» como el `xflip` de ADA). Entonces $\sum_\tau\chi(\tau)=0$ para
  los caracteres no triviales y $\hat w=(1-p)+p\cdot 0=1-p$.
  **Nuestro caso es distinto:** volteamos **solo los reales**, a nivel de dataset (`flip_pairs`, $p=0.5$, ambas
  vistas a la vez). Eso no es una augmentación dentro de $D$: cambia deliberadamente el objetivo a
  $\tilde p_d=\tfrac12(p_d+F_\#p_d)$, así que $G$ aprenderá a generar ambas orientaciones (lo aceptamos: duplica los
  datos efectivos y el par sigue siendo coherente, es la criatura espejada).
- **Cutout** (cuadro de 48×48): no tenemos prueba de inyectividad; la evidencia es empírica (DiffAugment) y la
  verificamos mirando las muestras (no deben aparecer cuadros grises).
- **Color: NO.** Si el matiz se rota uniformemente con probabilidad $p$, $P(H'\mid c)=(1-p)P(H\mid c)+p\,\mathcal U$ y

  $$
  \mathrm{TV}\big(P(H'\mid c_1),P(H'\mid c_2)\big)=(1-p)\,\mathrm{TV}\big(P(H\mid c_1),P(H\mid c_2)\big):
  $$

  **la señal tipo→color se atenúa justo en la variable que condiciona**. Medimos que el color predice el tipo
  mucho mejor que la luminancia (AUC medio 0.658 vs. 0.574; Fuego 0.76 vs. 0.56). Además, la saturación oficial de
  DiffAugment ($\mathcal U(0,2)$) puede desaturar del todo y promedia sobre los 6 canales, mezclando frente y
  espalda. DiffAugment usó exactamente *translation + cutout* para BigGAN condicional.
- **Pareada:** el **mismo** $\varphi$ (mismo desplazamiento, misma máscara) para las dos vistas de un par. Si se
  augmentaran por separado, los reales que ve $D$ incluirían pares incoherentes y $D$ aprendería que la
  incoherencia es «real».

### 7.5 LeCam: regularizar a D con anclas

$$
R_{LC}=\mathbb E_x\big[C(x)-\alpha_F\big]_+^2+\mathbb E_z\big[\alpha_R-C(G(z))\big]_+^2,\qquad
\mathcal L_D\leftarrow\mathcal L_D+\lambda R_{LC},\ \ \lambda=0.3,
$$

con $\alpha_R,\alpha_F$ medias móviles (EMA 0.99) de los logits reales y falsos [Tseng et al. 2021, arXiv:2104.03310]. Acerca
los logits reales al ancla histórica de los falsos y viceversa: impide que $D$ separe ambas clases sin límite.

**Qué divergencia minimiza.** Con un crítico tipo WGAN y anclas $\alpha_R=\alpha$, $\alpha_F=-\alpha$, $D$ maximiza
punto a punto $p_d d-p_g d-\lambda p_d(d+\alpha)^2-\lambda p_g(d-\alpha)^2$. Derivando e igualando a 0:

$$
d^{\star}(x)=\frac{1-2\lambda\alpha}{2\lambda}\cdot\frac{p_d-p_g}{p_d+p_g},\qquad
\int(p_d-p_g)\,d^{\star}=\Big(\frac{1}{2\lambda}-\alpha\Big)\Delta(p_d\Vert p_g),\qquad
\Delta(P\Vert Q)=\int\frac{(P-Q)^2}{P+Q}.
$$

$\Delta$ es la **divergencia de LeCam** (discriminación triangular), una $f$-divergencia con $f(t)=(t-1)^2/(t+1)$,
acotada ($0\le\Delta\le 2$) y equivalente a la JSD: $\tfrac14\Delta\le\mathrm{JSD}\le\tfrac{\ln 2}{2}\Delta$. La clave para
pocos datos: $\lvert d^{\star}\rvert\le\frac{1-2\lambda\alpha}{2\lambda}$ **incluso donde $p_g=0$** (puntos memorizados),
mientras que el $C^{\star}=\log(p_d/p_g)$ de la GAN logística diverge ahí. Evidencia: BigGAN + DiffAugment con 10 %
de CIFAR-10 pasa de FID 23.34 a 16.69 con LeCam.

**¿Sobre qué logit se aplica? (lección de nuestra revisión de código).** Nuestro $D$ tiene dos logits entrenados
por la hinge: $C_{uc}=\psi$ y $C_c=\psi+\langle Vy,\phi\rangle$. La primera versión aplicaba LeCam solo a $C_c$. Pero
entre las iteraciones 1k y $T_s$ ($\lambda_t=0$) el **único** gradiente que recibe $V$ es el de LeCam, que empuja
$C_c$ de los reales hacia abajo y el de los falsos hacia arriba; como $C_{uc}$ ya separa reales de falsos, $D$
satisface LeCam volviendo la proyección **anti-discriminativa** ($\langle Vy,\phi\rangle$ real < falso; en un
experimento de juguete la separación de la proyección quedó en $-4.6$). Con ese $V$ invertido, al encender
$\lambda_t$ el término condicional empujaría a $G$ en la dirección equivocada. **Corrección:** regularizar los
mismos logits que entrena la hinge y con el mismo peso, con un par de anclas para cada uno:

$$
R=R_{LC}(C_{uc};\alpha^{uc})+\lambda_t\,R_{LC}(C_c;\alpha^{c}).
$$

Moraleja: un regularizador debe actuar sobre la misma cantidad que optimiza la pérdida; si
se aplica a una suma de términos, la red puede «pagarlo» con el término que no tiene otra presión.

### 7.6 Muestreo ponderado

Muestreamos los reales con $w_i\propto n_{\text{sp}(i)}^{-0.5}\cdot m_{c_i}^{-0.25}\cdot(0.5\ \text{si shiny})$, donde
$n_{\text{sp}}$ son los pares de la especie y $m_c$ las especies distintas de la combinación. **Lema:** si $w$
dependiera solo de $c$, $q_w(x\mid c)=q(x\mid c)$ (solo cambia cuánto peso recibe cada condición); como depende de la
especie, $q(x\mid c)$ se vuelve más uniforme **sobre especies**: aprendemos «criaturas», no «sprites repetidos»
(Unown, Pikachu). Los falsos usan las mismas etiquetas, así que $p_g(y)=q_w(y)$ y la §4.1 sigue valiendo.

**En nuestro código:** [`augment.py`](../src/pokegan/augment.py): `diff_augment(x, translation=0.125, cutout=0.5)` (un
sorteo por muestra compartido por los 6 canales, relleno $+1$ en la traslación) y `flip_pairs(x, p=0.5)` (solo
reales); [`losses.py`](../src/pokegan/losses.py): `LeCam(decay=0.99)` con `.update` y `.penalty`;
[`data.py`](../src/pokegan/data.py): `sampling_weights`; `r_t` se registra en `runs/<nombre>/log.csv`.

---

## 8. Colapso por condicionamiento y entrenamiento transicional

**El fenómeno.** Shahbazi et al. [2022, arXiv:2201.06578] muestran que, con pocos datos, **condicionar por clase
causa colapso de modo** mientras que el mismo modelo incondicional no colapsa; lo condicional solo gana por encima
de ~5k imágenes. Intuición: el término condicional parte los datos en muchos problemas pequeños (en nuestro caso
160 combinaciones con una mediana de pocas especies cada una); $D$ sobreajusta por condición mucho antes que en la
marginal, y $G$ recibe gradientes que apuntan a unos pocos ejemplos por condición.

**El remedio: empezar incondicional y meter el tipo de a poco.**

$$
\lambda_t=\mathrm{clip}\Big(\frac{t-T_s}{T_e-T_s},\,0,\,1\Big),\qquad T_s=0.1\,T,\quad T_e=0.4\,T,
$$

y $\lambda_t$ entra en **dos** lugares:

$$
G:\ \hat e=\lambda_t\,\mathrm{RMSNorm}(E^\top y);\qquad
\mathcal L_D=H(C_{uc})+\lambda_t\,H(C_c),\qquad
\mathcal L_G=-\mathbb E\big[C_{uc}\big]-\lambda_t\,\mathbb E\big[C_c\big],
$$

con $C_{uc}=\psi(\phi)$ (estima $\log p_d(x)/p_g(x)$), $C_c=\psi(\phi)+\langle Vy,\phi\rangle$ (estima el cociente
condicional) y

$$
H(C)=\mathbb E_{\text{real}}\big[1-C\big]_+ +\mathbb E_{\text{falso}}\big[1+C\big]_+ +\tfrac12\,\mathbb E_{i}\big[1-C(f_i,b_i)+C(f_i,b_j)\big]_+ .
$$

(término desajustado por pares; ver §6.x).

- Con $\lambda_t=0$, $G$ **ignora** $y$ y aprende la marginal $p_d(x)$ con todos los datos: qué es un sprite, el
  contorno, el fondo blanco, la coherencia frente/espalda.
- Al subir $\lambda_t$, el objetivo es una **homotopía** (continuación) del problema incondicional al
  condicional: el conocimiento del tipo se agrega sobre una solución que ya cubre los modos.
- Con $\lambda_t=1$ siguen entrenándose ambas cabezas; la incondicional queda como término auxiliar.

**Evidencia (y su alcance).** En la ablación del paper (Food101): transición solo en la pérdida, FID 80 (no ayuda);
solo en $G$, 25; completa, 20; empezar en $T_s=0$ empeora (24). Se midió en StyleGAN2-ADA con ~2k imágenes y una
etiqueta por imagen: en BigGAN-CBN con multi-etiqueta **no está verificado**. Lo presentamos como un seguro barato,
no como una ganancia demostrada.

**Vigilar la diversidad.** El batch de $G$ usa etiquetas duplicadas en dos mitades con $z$ distintos, así que
registramos gratis $\mathrm{div}=\overline{\lvert G(z_1,y)-G(z_2,y)\rvert}$, el numerador de la razón de MSGAN
[Mao et al. 2019, arXiv:1903.05628] (el denominador $\overline{\lvert z_1-z_2\rvert}$ es casi constante). Si cae hacia 0
(colapso), se puede activar $\mathcal L_{ms}=1/\big(\overline{\lvert G(z_1,y)-G(z_2,y)\rvert}/\overline{\lvert z_1-z_2\rvert}+10^{-5}\big)$
(peso 0 por defecto).

**En nuestro código:** [`losses.py`](../src/pokegan/losses.py): `transition_lambda(step, t_start, t_end)` y
`mode_seeking(x1, x2, z1, z2)`; [`models/generator.py`](../src/pokegan/models/generator.py): el argumento `lam` de
`forward`; [`train.py`](../src/pokegan/train.py) guarda `lam` y `div` en `log.csv` y `lam` en el checkpoint.

---

## 9. Optimización: TTUR, Adam con β1 = 0, EMA y standing statistics

### 9.1 La dinámica de un juego rota (Dirac-GAN)

El ejemplo mínimo [Mescheder et al. 2018, arXiv:1801.04406]: $p_d=\delta_0$, $p_\theta=\delta_\theta$, $C_\psi(x)=\psi x$ y
$L(\theta,\psi)=f(\psi\theta)$ con $f$ cóncava, $f'(0)\ne 0$. El único equilibrio es $(0,0)$ y el jacobiano del campo
de gradiente ahí es

```math
v'(0,0)=\begin{pmatrix}0 & -f'(0)\\ f'(0) & 0\end{pmatrix}\quad\Rightarrow\quad\lambda=\pm\, i\,f'(0).
```

Autovalores imaginarios puros: **la dinámica gira** alrededor del equilibrio. En tiempo continuo
$\theta^2+\psi^2$ se conserva; con descenso simultáneo, $\lvert\lambda\rvert=\sqrt{1+h^2f'(0)^2} > 1$ y diverge. Una
penalización centrada en cero (R1) actúa como **fricción** ($\lambda=-\gamma/2\pm\sqrt{\gamma^2/4-f'(0)^2}$); nosotros
atacamos el mismo problema con SN, TTUR, $\beta_1=0$ y EMA.

### 9.2 TTUR y Adam

Heusel et al. [2017, arXiv:1706.08500] prueban, con aproximación estocástica en dos escalas de tiempo
($a(n)=o(b(n))$), que $D$ sigue casi su mejor respuesta y el par converge a un equilibrio de Nash local. En la
práctica: $\eta_D=4\cdot10^{-4}$, $\eta_G=10^{-4}$, un paso de $D$ por paso de $G$ (SAGAN; BigGAN encontró estos
valores óptimos con batch pequeño). Nuestra razón constante 4:1 es la versión práctica, no literalmente
$a(n)=o(b(n))$.

**Adam con $\beta=(0,\,0.999)$, eps $10^{-6}$.** En un juego que gira, el momento acumula la componente rotacional
y produce sobrepasos; BigGAN reporta que un $\beta_1$ mayor «arruinaba el entrenamiento».

### 9.3 EMA del generador

$$
\bar\theta_t=\beta\,\bar\theta_{t-1}+(1-\beta)\,\theta_t,\qquad\beta=0.999\ \text{desde el paso 1000},\qquad
n_{1/2}=\frac{\ln 0.5}{\ln\beta}\approx 693\ \text{pasos}.
$$

(Con $\beta=0.9999$, $n_{1/2}\approx 6931$: demasiado lento para unas decenas de miles de pasos.) **Por qué ayuda:** si
los iterados giran, $\theta_t=\theta^{\star}+r\,e^{i\omega t}$ (en el plano del giro), el promedio exponencial queda en

$$
\bar\theta_t-\theta^{\star}=r\,e^{i\omega t}\,\frac{1-\beta}{1-\beta e^{-i\omega}},\qquad
\Big\lvert\frac{1-\beta}{1-\beta e^{-i\omega}}\Big\rvert\approx\frac{1-\beta}{\omega}\ \ \text{si}\ \omega\gg 1-\beta,
$$

por ejemplo 0.1 para $\omega=0.01$ rad/paso: **el promedio cae cerca del centro de la órbita**
[Yazici et al. 2018, arXiv:1806.04498]. Todas las muestras, figuras y métricas usan $G_{\text{ema}}$.

### 9.4 Standing statistics (por qué no basta con promediar)

$G$ tiene BatchNorm. En `eval` usa medias y varianzas acumuladas, pero las de los pesos promediados **no** son el
promedio de las de cada paso: $\mu_\ell(\bar\theta)=\mathbb E_{z,y}\big[h_\ell(z,y;\bar\theta)\big]\ne\sum_t w_t\,\mu_\ell(\theta_t)$ porque
$h_\ell$ es no lineal en $\theta$. Con estadísticas viejas las muestras salen lavadas o saturadas. Remedio (BigGAN,
apéndice C): antes de muestrear, recalcular las estadísticas con varias pasadas en modo `train`, `momentum=None`
(promedio acumulado), $z\sim\mathcal N(0,I)$ e $y$ tomado del banco de etiquetas de entrenamiento, y volver a `eval`.

**En nuestro código:** [`ema.py`](../src/pokegan/ema.py): `EMA(model, decay=0.999, start=1000)` (promedia los parámetros y copia los buffers) y
`standing_stats(G, ys, n_batches=32, batch=64, lam=1.0)`; [`train.py`](../src/pokegan/train.py) configura Adam (TTUR) y
guarda `G_ema` y `label_bank` en el checkpoint.

---

## 10. Evaluación

Los modelos **preentrenados se usan solo para evaluar** (Inception para KID/FID) y nunca dentro de $G$, $D$ o sus
pérdidas. El clasificador de tipos se entrena **desde cero**.

| Requisito | Pregunta | Métrica |
|---|---|---|
| Criaturas realistas | $p_g(x)\approx p_d(x)$ | KID (titular) y FID, por vista |
| Condicionadas por tipo | $p_g(x\mid y)\approx p_d(x\mid y)$ | mAP / F1 / acierto de conjunto del clasificador de tipos |
| Frente y espalda coherentes | conjunta, no marginales | intersección de histogramas de paleta |
| Distintos $z$ ⇒ distintas criaturas | $\mathrm{Var}_z\,G(z,y)\gg 0$ | distancia media entre muestras de la misma condición |
| Pokémon **nuevos** | $G(z,y)\notin\mathcal D_{\text{train}}$ | distancia al vecino más cercano vs. especies reservadas |

### 10.1 FID: distancia de Fréchet entre gaussianas

Se ajusta una gaussiana a las características de Inception de cada conjunto. La distancia de Wasserstein-2 entre
gaussianas tiene forma cerrada [Heusel et al. 2017, arXiv:1706.08500]:

$$
\mathrm{FID}=\lVert\mu_r-\mu_g\rVert_2^2+\mathrm{Tr}\Big(\Sigma_r+\Sigma_g-2\big(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\big)^{1/2}\Big).
$$

El primer término capta un sesgo global (de color, por ejemplo); el segundo, la dispersión: si $G$ colapsa
($\Sigma_g\to 0$), tiende a $\mathrm{Tr}\,\Sigma_r$. **Sesgo:** $\hat\mu,\hat\Sigma$ son estimadores, y un desarrollo de
Taylor da $\mathbb E[\mathrm{FID}_N]=\mathrm{FID}_\infty+K_g/N+O(N^{-2})$ con $K_g$ **dependiente del generador**
[Chong y Forsyth 2019, arXiv:1911.07023]. Con $p_g=p_d$ el FID estimado sigue siendo positivo, no existe un estimador
insesgado [Bińkowski et al. 2018, arXiv:1801.01401], y con $d=2048 > N$ (por tipo) la covarianza es singular. Por eso el
FID solo se reporta como referencia.

### 10.2 KID: MMD² insesgada con kernel cúbico

$$
\mathrm{MMD}^2(P,Q)=\mathbb E\,k(x,x')+\mathbb E\,k(y,y')-2\,\mathbb E\,k(x,y),\qquad k(x,y)=\Big(\tfrac1d x^\top y+1\Big)^3 .
$$

Expandiendo con $s=x^\top y/d$, $(1+s)^3=1+3s+3s^2+s^3$: compara momentos de orden 1, 2 y 3 (media, covarianza,
asimetría) sin suponer gaussianidad. Su estimador U

$$
\widehat{\mathrm{MMD}}^2_u=\frac{1}{m(m-1)}\sum_{i\ne j}k(x_i,x_j)+\frac{1}{n(n-1)}\sum_{i\ne j}k(y_i,y_j)-\frac{2}{mn}\sum_{i,j}k(x_i,y_j)
$$

es **insesgado** (puede salir levemente negativo cuando las distribuciones coinciden, y es normal). Se promedia sobre
subconjuntos y, al no tener sesgo, **es comparable entre distintos $N$**: es nuestra métrica titular, por vista
(frente y espalda por separado). Preprocesado idéntico para reales y generadas: cuantizar a `uint8` y el mismo
reescalado [Parmar et al. 2021, arXiv:2104.11222].

### 10.3 Clasificador de tipos: techo y piso

Un ResNet-18 (timm, `pretrained=False`) multi-etiqueta entrenado con BCE sobre especies de entrenamiento, aplicado a
las muestras generadas (*GAN-test* [Shmelkov et al. 2018, arXiv:1807.09499]). Reportamos

- $\mathrm{AP}_t=\sum_n(R_n-R_{n-1})P_n$ y $\mathrm{mAP}=\frac1{18}\sum_t\mathrm{AP}_t$;
- F1 y **acierto de conjunto**: los $\lvert c\rvert$ tipos más probables son exactamente $c$ (azar ≈ 1/18 en monotipos y
  1/153 en duales);

siempre entre un **techo** (el mismo clasificador sobre reales de especies reservadas) y un **piso** (etiquetas
barajadas), con la normalización

$$
s=\frac{m_{\text{gen}}-m_{\text{piso}}}{m_{\text{techo}}-m_{\text{piso}}}.
$$

$s > 1$ no significa «mejor que lo real»: indica muestras estereotipadas (como las de AC-GAN). El clasificador
**nunca** participa en el entrenamiento (sería «enseñar para la prueba»).

### 10.4 Coherencia frente/espalda: intersección de histogramas de paleta

Cuantizamos RGB a 8 niveles por canal (512 celdas), excluimos el fondo blanco y normalizamos:

$$
\mathrm{HI}(x_f,x_b)=\sum_{b=1}^{512}\min\big(h_f(b),h_b(b)\big)\in[0,1],
$$

la fracción de «masa de color» compartida por las dos vistas. No requiere entrenamiento y está calibrada en pares
reales: **emparejados 0.706**, misma forma con paleta shiny cambiada 0.246, **espalda de otra especie 0.139**
(AUC 0.988). Se calcula sobre la salida **cruda** (el post-procesado con paleta conjunta la inflaría).

### 10.5 Diversidad y novedad

- **Diversidad intra-condición:** distancia media entre pares de muestras $G(z_i,y),G(z_j,y)$ (distancia coseno entre
  características de Inception y L1 en píxeles), y el puntaje Vendi $\exp\big(-\sum_k\lambda_k\log\lambda_k\big)$ con
  $\lambda_k$ los autovalores de la matriz de similitud normalizada: el «número efectivo de criaturas distintas»
  [Friedman y Dieng 2022, arXiv:2210.02410]. Se compara con la misma cantidad sobre reales de **especies distintas**;
  una razón muy por debajo de 1 indica colapso.
- **Novedad (no copia):** $d_{\mathrm{NN}}(x)=\min_{x'\in\mathcal D_{\text{train}}}\lVert\phi(x)-\phi(x')\rVert$ (en nuestro código,
  $\phi$ = píxeles y distancia L2). La
  referencia correcta son los Pokémon **reales nunca vistos** (especies reservadas): si las generadas están tan lejos
  del entrenamiento como ellos, son tan «nuevas» como un Pokémon real nuevo. Se compara la distribución (por ejemplo
  con Mann–Whitney) y se muestran las muestras junto a sus vecinos más cercanos [Meehan et al. 2020, arXiv:2004.05675].
- **Combinaciones no vistas:** se evalúan aparte (¿el clasificador pone ambos tipos en su top-2?).

**En nuestro código:** [`evaluate.py`](../src/pokegan/evaluate.py): `kid_fid` (KID y FID con la Inception de
`torch-fidelity`), `palette_hi`, `type_metrics`, `diversity` (coseno, Vendi, L1), `nearest_neighbors` y `nn_figure`;
`quick_eval` es la versión rápida que corre cada 5000 pasos durante el entrenamiento y elige `best.pt` por el KID
medio. [`type_classifier.py`](../src/pokegan/type_classifier.py): `train_type_classifier` (ResNet-18 de `timm` con
`pretrained=False`, BCE con `pos_weight`, aumentos solo geométricos). [`scripts/evaluate.py`](../scripts/evaluate.py)
escribe `metrics.json`, `metrics.md` (tablas con techo y piso) y las figuras de novedad.

---

## 10b. Muestreo guiado por D: el GAN es secretamente un modelo de energía

Al terminar el entrenamiento **no tiramos D**: su logit sabe dónde está la distribución real. Esta sección deriva cómo usarlo para muestrear mejor **sin reentrenar** (solo G y D propios, entrenados desde cero). Referencias: DDLS [Che et al. 2020, arXiv:2003.06060], DOT [Tanaka 2019, arXiv:1910.06832], DRS [Azadi et al. 2019, arXiv:1810.06758], MH-GAN [Turner et al. 2019, arXiv:1811.11357], DGflow [Ansari et al. 2021, arXiv:2012.00780], truncamiento [Brock et al. 2018, arXiv:1809.11096].

**1. GAN como modelo de energía (Che et al. 2020).**
Con pérdida logística y G fijo, el D óptimo es $D^*(x)=\frac{p_d(x)}{p_d(x)+p_g(x)}$. Sea $d=\operatorname{logit}D^*$:
$$\sigma(d)=\frac{1}{1+p_g/p_d}\;\Rightarrow\; e^{d(x)}=\frac{p_d(x)}{p_g(x)}\;\Rightarrow\; p_d(x)=p_g(x)\,e^{d(x)}.$$
En la práctica se define $p^*_d(x)=p_g(x)\,e^{d(x)}/Z_0$, con $Z_0=\mathbb E_{p_g}[e^{d}]$ (vale 1 en el óptimo).

**2. Pullback al latente: no hace falta el jacobiano.**
Sea $p_t(z)=p_0(z)\,e^{d(G(z))}/Z$. Para cualquier conjunto $A$:
$$\Pr_{z\sim p_t}[G(z)\in A]=\frac1Z\int \mathbf 1_A(G(z))\,e^{d(G(z))}\,p_0(z)\,dz=\frac1Z\,\mathbb E_{x\sim p_g}\!\big[\mathbf 1_A(x)\,e^{d(x)}\big]=\int_A \frac{p_g(x)\,e^{d(x)}}{Z}\,dx .$$
Por tanto $G_\#p_t=p^*_d$. El peso depende de $z$ solo a través de $G(z)$, así que la reponderación pasa por el push-forward sin determinantes. Con $p_0=\mathcal N(0,I_{120})$:
$$E(z)=\tfrac12\|z\|^2-d(G(z)),\qquad p_t\propto e^{-E}.$$

**3. Langevin y el papel de $N$.**
Discretización de Euler–Maruyama de $dz=-\tfrac12\nabla E\,dt+dW$:
$$z_{i+1}=z_i-\tfrac{\varepsilon}{2}\nabla_z E(z_i)+\sqrt{\varepsilon}\,\xi_i=\Big(1-\tfrac{\varepsilon}{2}\Big)z_i+\tfrac{\varepsilon}{2}\nabla_z d(G(z_i))+\sqrt{\varepsilon}\,\xi_i .$$
Solo con el prior es un AR(1):
- $\operatorname{Corr}(z_N,z_0)=(1-\varepsilon/2)^N\approx e^{-\varepsilon N/2}$.
- Varianza estacionaria $\varepsilon/\big(1-(1-\varepsilon/2)^2\big)\approx 1+\varepsilon/4$ (sesgo de discretización pequeño).

Con $\varepsilon=0.01$: $N=200$ da $e^{-1}\approx0.37$ (refina el mismo individuo) y $N=1000$ da $e^{-5}\approx0.007$ (muestra nueva). El calendario de ImageNet ($\varepsilon\times0.1$ cada 200 pasos) acumula un tiempo total $\sum_i\varepsilon_i/2\approx1.1$: es esencialmente local.

**4. Bisagra: el logit no es un log-cociente.**
La pérdida puntual es $\ell(C)=p_d(x)\,(1-C)_++p_g(x)\,(1+C)_+$. En $[-1,1]$ vale $p_d(1-C)+p_g(1+C)$, lineal con pendiente $p_g-p_d$, y fuera de ese intervalo solo crece. Por tanto
$$C^*(x)=\operatorname{sign}\big(p_d(x)-p_g(x)\big)=\operatorname{sign}\,\log r(x),\qquad r=p_d/p_g .$$
El óptimo de la bisagra solo conserva el signo. Con SN, $\|C\|_{\rm Lip}\le K$, así que el $C$ real es un signo suavizado: monótono en $\log r$ cerca de la frontera y saturado lejos de ella.

Modelo de trabajo: $\log r(x)\approx\beta\,C(x)+b$. Calibración de Platt con $n$ reales de validación ($y_i=1$) y $n$ falsos con las mismas etiquetas ($y_i=0$):
$$(\hat\beta,\hat b)=\arg\min_{\beta,b}\;-\sum_i\Big[y_i\log\sigma(\beta s_i+b)+(1-y_i)\log\sigma(-\beta s_i-b)\Big].$$
- Con clases balanceadas, $\hat\beta s+\hat b$ estima $\log r$; si no, se resta $\log(n_1/n_0)$.
- $b$ no importa en Langevin, DOT ni MH, y en DRS queda absorbido por $d_M$.
- LeCam penaliza $\mathbb E[(C_{\rm real}-\alpha_F)_+^2]+\mathbb E[(\alpha_R-C_{\rm falso})_+^2]$, que acerca cada logit al ancla contraria y comprime el rango. Por eso se espera $\hat\beta>1$: el logit crudo está "enfriado".

**5. Caso condicional y proyección.**
Como $p_g(y)=p_d(y)$ (train.py toma $y_g$ de las etiquetas reales):
$$\log\frac{p_d(x\mid y)}{p_g(x\mid y)}=\underbrace{\log\frac{p_d(x)}{p_g(x)}}_{\leftrightarrow\;C_{uc}}+\underbrace{\log\frac{p_d(y\mid x)}{p_g(y\mid x)}}_{\leftrightarrow\;\langle Vy,\phi(x)\rangle=\sum_{t\in y}\langle v_t,\phi\rangle}.$$
El objetivo condicional es $p_d(x|y)\propto p_g(x|y)\,e^{\beta C_c}$, con $C_c=C_{uc}+\mathrm{proj}$. Inclinación con guía:
$$\tilde p_w(x\mid y)\propto p_g(x\mid y)\;r(x)^{\beta}\Big[\tfrac{p_d(y|x)}{p_g(y|x)}\Big]^{\beta w},\qquad d_w=\beta\,(C_{uc}+w\,\mathrm{proj}).$$
- Con $w>1$ se refuerza la evidencia de tipo, igual que la escala de la guía por clasificador.
- La pérdida de G fue $-C_{uc}-C_c=-2\,(C_{uc}+\tfrac12\mathrm{proj})$: G se entrenó con $w=\tfrac12$ implícito.
- Los negativos desajustados (frente$_i$, espalda$_j$) hacen que $C$ también mida la coherencia entre vistas.

**6. DiffAugment: qué estima D y por qué promediar.**
D ve $u=T(x)$ con $T\sim\mathcal T$: traslación de $\pm12$ px con relleno blanco y cutout de $48\times48$ a 0 con $p=1$. Su óptimo compara $T_\#p_d$ con $T_\#p_g$.
- Traslación: si el sprite no se sale del lienzo, $\log r_T(Tx)=\log r(x)$.
- Cutout con máscara $m$: $(T_{m\#}p)(m\odot x)=p(x_{{\rm vis}(m)})$, la marginal de los píxeles visibles. Entonces $\log r_{T_m}(m\odot x)=\log\frac{p_d(x_{\rm vis})}{p_g(x_{\rm vis})}$.

Promediar da una razón de **verosimilitud compuesta** (pseudo-verosimilitud):
$$\bar s(x)=\mathbb E_T\big[C(Tx)\big]\;\approx\;\beta\;\mathbb E_m\Big[\log\tfrac{p_d(x_{{\rm vis}(m)})}{p_g(x_{{\rm vis}(m)})}\Big]+b .$$
Es además exactamente lo que maximizó G: $\mathbb E_T\,C(T\,G(z))$. Con $K$ aumentos el estimador es insesgado y su varianza es $\sigma_T^2/K$. En Langevin eso da un gradiente estocástico (tipo SGLD), y con $\varepsilon$ pequeño basta $K=4$. Para rankear (DRS, MH) conviene $K\ge16$. Una imagen limpia, sin cutout, nunca la vio D.

**7. DOT (Tanaka 2019).**
Dualidad de Kantorovich–Rubinstein:
$$W_1(p,q)=\min_{\pi\in\Pi(p,q)}\mathbb E_\pi\|x-y\|_2=\max_{\|f\|_{\rm Lip}\le1}\ \mathbb E_p f-\mathbb E_q f .$$
Sean $(\pi^*,f^*)$ óptimos y $T$ el mapa de Monge de $q$ a $p$. Por holgura complementaria, $f^*(T(y))-f^*(y)=\|T(y)-y\|$ casi seguro; por ser Lipschitz, $f^*(x')-f^*(y)\le\|x'-y\|$ para todo $x'$. Restando:
$$\|x'-y\|-f^*(x')\;\ge\;-f^*(y)\;=\;\|T(y)-y\|-f^*(T(y))\quad\Rightarrow\quad T(y)\in\arg\min_x\{\|x-y\|-f^*(x)\}.$$
El argmin no es único (errata de 2023): en el segmento $x_t=t\,y+(1-t)\,T(y)$, $f^*$ crece con pendiente 1, así que $\|x_t-y\|-f^*(x_t)$ es constante. El descenso por gradiente puede detenerse a mitad de camino.

*Bisagra, SN y dualidad.* Con $V_D=\mathbb E_p[\min(0,-1+D)]+\mathbb E_{p_G}[\min(0,-1-D)]$ y $\min(0,a)\le a$:
$$V_D\le\mathbb E_pD-\mathbb E_{p_G}D-2\le K\big(\mathbb E_p\tilde D-\mathbb E_{p_G}\tilde D\big)\le K\,W_1(p,p_G),\qquad \tilde D=D/K .$$
Maximizar la bisagra sube una cota inferior del dual, de modo que $\tilde D\approx f^*$. SN garantiza $\|D\|_{\rm Lip}\le\prod_l\|W_l/\sigma(W_l)\|=1$, pero esa cota es muy holgada: en SNGAN sobre CIFAR la constante efectiva ronda 0.08. Por eso se usan las estimaciones empíricas
$$K_{\rm eff}=\max_{x,y\sim p_G}\frac{|D(x)-D(y)|}{\|x-y\|},\qquad k_{\rm eff}=\max_{z,z'}\frac{|D(G(z))-D(G(z'))|}{\|z-z'\|}.$$
En la versión latente, $D\circ G$ hace de potencial dual entre $p_Z$ y el pullback de $p$:
$$T(z_y)\in\arg\min_z\Big\{\|z-z_y\|_2-\tfrac{1}{k_{\rm eff}}\,\bar s\big(G(z,y),y\big)\Big\}.$$
El gradiente del costo, $(z-z_y)/\|z-z_y\|$, tiene norma 1. Solo hay movimiento neto donde $\|\nabla_z\bar s\|/k_{\rm eff}>1$, es decir donde D es más empinado que su pendiente secante típica.

*Proyección tangente.* Bajo $\mathcal N(0,I_D)$ la masa vive cerca de $\|z\|=\sqrt D$. La proyección exacta sobre el plano tangente es
$$P g=g-\frac{(g\cdot z)}{\|z\|^2}\,z .$$
El paper y el código usan $g-\frac{(g\cdot z)}{\sqrt D}z$. Su componente radial es $g_r\big(1-\|z\|^2/\sqrt D\big)\approx g_r(1-\sqrt D)$: con $D=120$ la invierte y la multiplica por $\approx 9.95$. Para reproducir el paper sirve su fórmula; para la geometría correcta, usar $\|z\|^2$.

**8. DRS (Azadi et al. 2019).**
Rechazo con propuesta $p_g$: se acepta con probabilidad $\frac{p_d}{M\,p_g}=e^{d(x)-d_M}$, con $d_M=\log M$. Se busca $F$ tal que $\sigma(F)=e^{\Delta}$, con $\Delta=d-d_M\le0$:
$$1+e^{-F}=e^{-\Delta}\;\Rightarrow\;e^{-F}=e^{-\Delta}\,(1-e^{\Delta})\;\Rightarrow\;F=\Delta-\log\!\big(1-e^{\Delta}\big).$$
En la práctica, $\hat F=\Delta-\log(1-e^{\Delta-\epsilon})-\gamma$ y $p_{\rm acc}=\sigma(\hat F)$.

*Lectura asintótica.* Si $\Delta\ll0$, entonces $-\log(1-e^{\Delta})\approx e^{\Delta}\to0$, así que $\hat F\approx\Delta-\gamma$ y
$$p_{\rm acc}(x)\approx\sigma\big(d(x)-d_M-\gamma\big)\;\approx\;e^{\,d(x)-(d_M+\gamma)}\quad\text{(lejos del umbral)}.$$
- Equivale a rechazo con $M_{\rm eff}=M e^{\gamma}<M$, con la aceptación recortada a 1 para los mejores; por eso deja de ser exacto.
- Con $\gamma=Q_p(\hat F)\approx Q_p(d)-d_M$:
$$p_{\rm acc}(x)\approx\sigma\big(d(x)-Q_p(d)\big)=\sigma\big(\beta\,(\bar s(x)-Q_p(\bar s))\big),$$
un umbral logístico suave en el percentil $p$.
- La tasa de aceptación es $R(\beta)=\mathbb E\,\sigma\big(\beta(\bar s-Q_p)\big)$: tiende a $1-p$ si $\beta\to\infty$ (top duro) y a $\tfrac12$ si $\beta\to0$ (azar). $\beta$ fija la dureza, y por eso hay que calibrar.
- Tomar $\gamma$ por condición $y$ preserva $p(y)$.

**9. MH-GAN (Turner et al. 2019).**
Muestreador de independencia con propuesta $q=p_g$ y objetivo $p_d$:
$$\alpha(x',x)=\min\Big(1,\frac{p_d(x')\,p_g(x)}{p_d(x)\,p_g(x')}\Big)=\min\Big(1,\frac{D(x)^{-1}-1}{D(x')^{-1}-1}\Big)=\min\big(1,e^{\,d(x')-d(x)}\big),$$
porque $D^{-1}-1=e^{-d}$. Con nuestra bisagra: $\alpha=\min\big(1,e^{\beta(\bar s(x')-\bar s(x))}\big)$. Cuando $K\to\infty$ equivale a remuestreo por importancia: entre $n$ propuestas se elige $x_i$ con probabilidad $\propto e^{\beta\bar s(x_i)}$.

**10. Una sola familia: el truncamiento es la temperatura del prior.**
$$E_{\sigma,\beta,w}(z)=\sum_{l=0}^{5}\frac{\|z_l\|^2}{2\sigma_l^2}\;-\;\beta\,\bar s_w\big(G(z,y),y\big),\qquad z=[z_0,\dots,z_5],\; z_l\in\mathbb R^{20}.$$
Casos particulares:
- $\beta=0$ y todos los $\sigma_l=\sigma$: truncamiento por escala (z_var de BigGAN-PyTorch).
- $\beta=0$ con soporte $[-\theta,\theta]^{20}$: truncamiento de BigGAN.
- $\sigma_{0..2}<1$ y $\sigma_{3..5}=1$: truncamiento solo en escalas gruesas (StyleGAN aplica $\psi=0.7$ solo en $4^2$–$32^2$).
- $\sigma=1$ y $\beta>0$: DDLS.
- Pocos pasos y sin término del prior: DGflow-KL. Como $f'(t)=\log t+1$ y $t=e^{-d}$, se tiene $\nabla f'(e^{-d})=-\nabla d$, y el paso es $z\leftarrow z+\eta\nabla_z d+\sqrt{2\eta\gamma}\,\xi$.
- El costo de transporte $\|z-z_y\|$ en lugar del prior: DOT.

Propuesta para siluetas: $\sigma_{0..2}\in[0.6,0.9]$ y $\beta\in[\hat\beta,2\hat\beta]$, moviendo solo $z_{0..2}$. Así se ataca la forma global y se conservan la paleta y el detalle ($z_{3..5}$). Es una hipótesis nuestra: validarla con KID, el conteo de componentes de $\alpha$ y la fracción de vecinos cercanos (novedad).

**En nuestro código:** [`refine.py`](../src/pokegan/refine.py): `d_logit` (C_uc + proyección promediado sobre DiffAugment), `ddls`, `dgflow` (con `chunks` para mover solo z0–z2), `dot`, `sir`, `truncate` (global o por trozos), `platt_beta` (calibración) y `count_fragments`; [`scripts/refine_eval.py`](../scripts/refine_eval.py) compara todos los métodos con las mismas semillas (KID, tipo, coherencia, diversidad, fragmentos, novedad).

---

## 10c. Disección del modelo entrenado y afinado de forma (co-área y soporte)

**Qué aprendió el latente.** Con índices de Sobol de efecto
total $S_{T,k}=\mathbb E\|\phi(A)-\phi(A_k)\|^2/(2\,\mathrm{Var}\,\phi)$ (re-muestrear solo el trozo $k$):
$S_T(z_0)=0.89$ en píxeles y $0.90$ en silueta, y $z_0$ concentra el 99.4 % de $\mathrm{tr}(J^\top J)$; $z_1..z_5$
casi no cambian la imagen. Motivo: $z_\ell$ solo entra por la BN condicional $\gamma=1+W_\gamma[z_\ell;\hat e]$, donde
su modulación por canal es ≈ 1/3 de la del tipo. **El modelo usa ~20 de sus 120 dimensiones.**

**El vacío es una dirección, no una cola.** Una dirección $\hat w$ de $z_0$ (coseno $-0.92$ con la de log-área) predice
los lienzos vacíos con AUC 0.88 usando solo $t=\langle z_0,\hat w\rangle$; la norma $\|z_0\|$ da AUC 0.50 (azar).
$P(\text{vacía}\mid t)\approx\sigma\big(3.5\,(t-t^*_{\text{tipo}})\big)$ y, como $t\sim\mathcal N(0,1)$, la tasa por tipo es
$1-\Phi(t^*)$: para Normal $t^*=0.35\Rightarrow 0.36$, igual a su tasa medida (33–41 %). Normal se vacía porque es
**pequeño**, no por pálido. El 85 % de los vacíos son RGB blanco con α encendido: en $x=1+\alpha(\mathrm{rgb}-1)$,
$\partial x/\partial\alpha=\mathrm{rgb}-1\approx0$, y $D$ los puntúa igual que un falso típico ⇒ no hay gradiente que los saque.

**Afinado de forma** (3000 pasos desde el paso 24 000, `configs/forma.yaml`). Dos regularizadores sobre $G$:

1. **Soporte (área visible).** Ningún sprite real tiene < 240 px de primer plano. Con primer plano suave
   $s=\sigma\big(50\,((1-\min_c x)-0.157)\big)$ medido sobre la imagen **compuesta** (no sobre α, que está encendido
   en los vacíos): $R_A=\big[240-\sum s\big]_+/240$. Es la versión entrenable de la restricción de soporte, que reduce
   $\mathrm{KL}(p_{data}\|p_g)$ en $-\log p_g(A)$ (§10b).
2. **Perímetro (co-área).** $\mathrm{TV}(\alpha)=\int|\nabla\alpha|\,dx=\int_0^1\mathrm{Per}(\{\alpha>t\})\,dt$: la variación total es el
   perímetro medio de los conjuntos de nivel. Las máscaras reales tienen TV ≈ 0.04 y las generadas ≈ 0.12 (3× más
   perímetro: bordes rotos y fragmentos; cada fragmento suelto suma perímetro). Bisagra hacia el nivel real:
   $R_P=[\mathrm{TV}(\alpha)-0.04]_+$.

$\mathcal L_G\leftarrow\mathcal L_G+2\,R_A+3\,R_P$. Resultado (mismas semillas):

| | KID×10³ frente/espalda | Acierto tipo | Fragmentos/vista | Cuerpo único | Vacíos (< 240 px) |
|---|---|---|---|---|---|
| paso 24 000 | 27.9 / 40.4 | 0.138 | 1.44 | 69 % | 8.8 % |
| **afinado 27 000** | **23.6** / 41.8 | **0.190** | **1.27** | **80 %** | **0.1 %** |

Lección: el análisis del latente y de $D$ (qué no ve $D$: fragmentos, vacíos) indicó **qué** regularizar; la
matemática (co-área, soporte) indicó **cómo** hacerlo sin pesos preentrenados.

**En nuestro código:** [`losses.py`](../src/pokegan/losses.py): `alpha_shape_penalty`; [`train.py`](../src/pokegan/train.py)
(pesos `alpha_area_weight`, `alpha_tv_weight`); [`models/generator.py`](../src/pokegan/models/generator.py)
(`return_alpha=True`); [`scripts/compare_ckpts.py`](../scripts/compare_ckpts.py).

---

## 11. Preguntas frecuentes

**1. ¿Por qué hinge y no la pérdida original?**
Con $D$ óptimo la original minimiza la JSD, que es constante cuando los soportes no se solapan (gradiente nulo); la
no saturante empuja hacia la KL inversa (colapso). La hinge con $D$ Lipschitz minimiza una métrica débil tipo Dudley,
entre $K\,W_1$ y $2\,\mathrm{TV}$, y da gradiente útil (§2).

**2. ¿Qué garantiza que frente y espalda sean la misma criatura?**
Tres cosas: un solo $z$ genera ambas vistas desde un tronco común (CoGAN); $D$ ve el par apilado, así que compara la
**conjunta** (dos $D$ separados solo verían marginales, §6.1); y los negativos desajustados le enseñan
explícitamente que un par incoherente es falso.

**3. ¿Los negativos desajustados no sesgan al generador?**
No, si están fuera del soporte de los datos: $D^{\star}=0$ ahí, su contribución al valor del juego es 0 y el objetivo
de $G$ queda en $2\,\mathrm{JSD}-\log 4$ exacto (§6.5). Por eso exigimos especie distinta y no usamos «tipo equivocado».

**4. ¿Cómo entra el tipo? ¿Por qué suma y no media?**
En $G$, por CBN en todas las escalas con $\hat e=\lambda_t\,\mathrm{RMSNorm}(E^\top y)$. En $D$, por proyección
$\langle Vy,\phi\rangle$. Si $p(c\mid x)$ es log-lineal sobre conjuntos, el logit óptimo es exactamente la suma; la media
solo cabe en la familia si todas las $v_t$ son iguales (§4.3–4.4).

**5. ¿Cómo genera una combinación que nunca vio, como Fuego/Hada?**
El multi-hot $\mathbf 1_{\text{fuego}}+\mathbf 1_{\text{hada}}$ define el embedding y el término de $D$ como suma de dos
expertos aprendidos con muchos datos. Una tabla de 171 combinaciones no tendría vector para ella (§4.5).

**6. ¿Por qué no AC-GAN?**
Su pérdida sobre falsos es $H_Q(Y\mid X)+\mathrm{KL}$: $G$ minimiza la entropía condicional y produce criaturas
estereotipadas, contra el requisito de diversidad (§4.6).

**7. Si augmentan, ¿por qué $G$ no genera imágenes trasladadas o con cuadros?**
Porque la augmentación se aplica a reales **y** falsos y es diferenciable: el juego compara $T_\#p_g$ con
$T_\#p_d$, y si $T_\#$ es inyectivo eso implica $p_g=p_d$. Para traslaciones lo probamos con caracteres:
$\hat w(k)\ne 0$ (§7.3–7.4).

**8. ¿Por qué no usan aumentos de color ni volteo dentro de D?**
El color es la señal del tipo: girar el matiz con probabilidad $p$ atenúa por $(1-p)$ la información tipo→color. Y un
volteo con $p=0.5$ dentro de $D$ tiene $\hat w=1-2p=0$ sobre la parte antisimétrica: filtra la orientación (§7.4).

**9. ¿Qué es la normalización espectral?**
Divide cada peso por su mayor valor singular, estimado con una iteración de potencia por paso. Acota la constante de
Lipschitz de $D$ por $\prod\sigma(W_l)$ (aproximadamente, por convoluciones y residuos) y eso convierte la hinge en
una métrica débil (§3).

**10. ¿Qué es $\lambda_t$ y por qué no condicionar desde el principio?**
Con pocos datos, condicionar causa colapso. $\lambda_t$ va de 0 a 1 entre el 10 % y el 40 % del entrenamiento y
multiplica el embedding en $G$ y el término condicional de la pérdida: primero se aprende «qué es un Pokémon» con
todos los datos y luego «de qué tipo» (§8).

**11. ¿Por qué EMA y standing statistics?**
La dinámica del juego gira alrededor del equilibrio; el promedio exponencial cae cerca del centro. Como $G$ tiene
BatchNorm, las estadísticas de los pesos promediados hay que recalcularlas antes de muestrear (§9).

**12. ¿Por qué KID y no solo FID?**
El FID está sesgado ($\mathbb E[\mathrm{FID}_N]=\mathrm{FID}_\infty+K_g/N$ con $K_g$ dependiente del modelo) y es
inestable con pocas muestras; el KID es una MMD² con estimador insesgado y comparable entre tamaños (§10.1–10.2).

**13. ¿Usaron modelos preentrenados?**
Solo para **evaluar**: Inception (KID/FID), declarado. $G$ y $D$ parten de pesos aleatorios, y ninguna pérdida de
entrenamiento usa una red preentrenada. El clasificador de tipos de la evaluación también se entrena desde cero.

**14. ¿Cómo saben que no copia Pokémon del dataset ni repite siempre la misma criatura?**
Novedad: comparamos la distancia de cada muestra a su vecino más cercano en entrenamiento con la de Pokémon reales de
**especies reservadas** (verdaderamente nuevos), y mostramos las muestras junto a sus vecinos (§10.5). Colapso:
grillas con $z$ fijos cada 500 pasos, la columna `div` de `log.csv` (con la pérdida de MSGAN de respaldo) y la
diversidad intra-condición en la evaluación.

**15. ¿Por qué quitaron las generaciones 1 y 2?**
El 37 % del dataset son sprites de Game Boy de 4 colores o en gris: mezclan estilos y debilitan la relación
tipo→color. Sin ellos conservamos las 1025 especies y 162 combinaciones, y el tamaño efectivo sube de 558 a 815. Es
curaduría de datos, declarada.
