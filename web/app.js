// PokeGAN en el navegador: generador ONNX + ONNX Runtime Web (WebGPU o WebAssembly).
"use strict";

const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const S = 96, PIX = S * S, BATCH = 4, WHITE = 235;   // fondo = casi blanco (min RGB ≥ 235) conectado al borde

const TYPE_COLORS = {
  normal: "#A8A77A", fighting: "#C22E28", flying: "#A98FF3", poison: "#A33EA1", ground: "#E2BF65", rock: "#B6A136",
  bug: "#A6B91A", ghost: "#735797", steel: "#B7B7CE", fire: "#EE8130", water: "#6390F0", grass: "#7AC74C",
  electric: "#F7D02C", psychic: "#F95587", ice: "#96D9D6", dragon: "#6F35FC", dark: "#705746", fairy: "#D685AD",
};
// las 11 combinaciones ausentes del conjunto de entrenamiento
const UNSEEN = new Set(["normal+rock", "normal+bug", "normal+steel", "normal+ice", "poison+ice", "ground+fairy",
  "rock+ghost", "bug+dragon", "bug+dark", "fire+fairy", "dragon+fairy"]);
// raíces para los nombres generados
const ROOTS = {
  normal: ["nor", "mim", "pel", "bal", "com"], fighting: ["pug", "brac", "fort", "lid", "pun"], flying: ["ala", "plum", "aer", "cel", "vent"],
  poison: ["tox", "ven", "mias", "bil", "cian"], ground: ["ter", "pol", "are", "gea", "lod"], rock: ["lit", "pedr", "gran", "mon", "bas"],
  bug: ["ins", "quit", "larv", "ant", "escar"], ghost: ["umbr", "espec", "fant", "nebl", "lem"], steel: ["ferr", "acer", "crom", "met", "tit"],
  fire: ["ign", "pyr", "bras", "flam", "cal"], water: ["aqu", "hidr", "mar", "ond", "nau"], grass: ["fol", "ram", "flor", "bot", "silv"],
  electric: ["volt", "chis", "elec", "ray", "tron"], psychic: ["psi", "ment", "onir", "aur", "esp"], ice: ["glac", "cri", "nev", "escarch", "friz"],
  dragon: ["drac", "wyr", "escam", "sierp", "draq"], dark: ["nox", "sombr", "nig", "ocul", "tenebr"], fairy: ["fey", "lum", "pix", "sil", "hal"],
};
const ENDS = ["on", "ix", "a", "ito", "or", "ello", "us", "ino", "ara", "ex", "ión", "ina", "uro", "eta"];
const RETRO = [[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]];
// íconos de tipo: glifos pixel art 9×9 originales (# = trazo sobre el color del tipo)
const GLYPHS = {
  normal:   ["...###...", ".#######.", ".##...##.", "##.....##", "##.....##", "##.....##", ".##...##.", ".#######.", "...###..."],
  fighting: [".........", ".##.##.#.", "#########", "#########", "#########", "########.", ".#######.", "..#####..", "..#####.."],
  flying:   [".........", "........#", "......###", "....#####", "..#######", "#########", ".########", "..#.#.#.#", "........."],
  poison:   [".....##..", "....#..#.", "....#..#.", ".##..##..", "#..#.....", "#..#..##.", ".##..#..#", ".....#..#", "......##."],
  ground:   [".........", "...###...", "..#####..", ".#######.", "#########", ".........", "#########", ".........", "#########"],
  rock:     ["...####..", "..######.", ".###.####", "###.#####", "##.######", "#########", "######.##", ".####.##.", "..######."],
  bug:      [".#.....#.", "..#...#..", "...###...", "..#####..", ".###.###.", ".###.###.", ".###.###.", "..##.##..", "...#.#..."],
  ghost:    ["..#####..", ".#######.", "##.###.##", "##.###.##", "#########", "#########", "#########", "#########", "#.#.#.#.#"],
  steel:    ["....#....", ".#.###.#.", "..#####..", ".##...##.", "###...###", ".##...##.", "..#####..", ".#.###.#.", "....#...."],
  fire:     ["....#....", "...##....", "...###...", "..####.#.", "..######.", ".###.###.", ".##...##.", ".##...##.", "..#####.."],
  water:    ["....#....", "....#....", "...###...", "..#####..", ".#######.", ".##.####.", ".##.####.", "..#####..", "...###..."],
  grass:    ["......###", "....#####", "...######", "..####.##", ".####.###", ".###.####", ".##.####.", ".#.###...", "#........"],
  electric: [".....###.", "....###..", "...###...", "..######.", ".....##..", "....##...", "...##....", "..##.....", ".#......."],
  psychic:  [".........", "..#####..", ".##...##.", "##.###.##", "#..###..#", "##.###.##", ".##...##.", "..#####..", "........."],
  ice:      ["....#....", ".#..#..#.", "..#.#.#..", "...###...", "#########", "...###...", "..#.#.#..", ".#..#..#.", "....#...."],
  dragon:   [".#.....#.", ".##...##.", "..#####..", ".#######.", "##.###.##", "#########", ".#######.", "..#.#.#..", "........."],
  dark:     ["...####..", "..###....", ".###.....", ".###.....", ".###.....", ".###.....", ".###.....", "..###....", "...####.."],
  fairy:    ["....#....", "....#....", "...###...", ".#######.", "#########", ".#######.", "...###...", "....#....", "....#...."],
};
// fuente pixel 5×7 para el logotipo
const FONT = {
  P: ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."], O: [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
  K: ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"], E: ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
  G: [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."], A: [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
  N: ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
};

const THEMES = [["lab", "Claro", "#22402f"], ["noche", "Nocturno", "#151c21"], ["retro", "Portátil", "#8bac0f"], ["tipo", "Del tipo", null]];
const STYLE_OPTS = [
  ["layout", "Vista", [["ficha", "Ficha"], ["combate", "Combate"], ["giro", "Giro"]]],
  ["bg", "Fondo", [["tipo", "Tipo"], ["escena", "Escena"], ["cuadricula", "Cuadrícula"], ["blanco", "Blanco"]]],
  ["scale", "Tamaño", [["2", "Pequeño"], ["3", "Mediano"], ["4", "Grande"]]],
  ["sprite", "Estilo", [["original", "Original"], ["contorno", "Contorno"], ["retro", "Retro"]]],
];
const MODEL_INFO = {
  vision: ["Vision", "Mejor anatomía. Entrenado con un discriminador auxiliar sobre ConvNeXt preentrenado; el generador es el mismo."],
  pura: ["Pura", "Generador y discriminador entrenados 100 % desde cero."],
};
const UI_DEFAULT = { theme: "lab", layout: "ficha", bg: "tipo", scale: "3", sprite: "original", shadow: "on" };

ort.env.wasm.wasmPaths = ORT_CDN;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const autotest = new URLSearchParams(location.search).has("autotest");
let META = null;
const SESSIONS = {};
let ui = { ...UI_DEFAULT };
try { ui = { ...UI_DEFAULT, ...JSON.parse(localStorage.getItem("pokegan-ui") || "{}") }; } catch {}
const st = { model: "vision", types: ["fire", "flying"], n: 6, fid: false, creatures: [], sel: 0, all18: false, seed: 0 };
let flipTimer = null;

// ---------- aleatoriedad reproducible ----------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hash(str) { let h = 2166136261; for (const ch of str) { h ^= ch.codePointAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
function randn(rng) { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }
// z ~ N(0, I); modo fidelidad: re-muestrea |z_i| > 1 en z0–z2 (los trozos de la forma global)
function makeZ(n, seed, fidelity) {
  const D = META.z_dim, chunk = D / META.n_chunks, z = new Float32Array(n * D);
  const rng = mulberry32((seed >>> 0) ^ 0x9E3779B9);
  for (let k = 0; k < n; k++) for (let i = 0; i < D; i++) {
    let v = randn(rng);
    if (fidelity && i < 3 * chunk) while (Math.abs(v) > 1) v = randn(rng);
    z[k * D + i] = v;
  }
  return z;
}
function oneHot(types) { const y = new Float32Array(18); for (const t of types) y[META.types.findIndex((p) => p[0] === t)] = 1; return y; }
const esName = (t) => META.types.find((p) => p[0] === t)[1];
const comboKey = (types) => [...types].sort((a, b) => META.types.findIndex((p) => p[0] === a) - META.types.findIndex((p) => p[0] === b)).join("+");

function creatureName(types, seed, k) {
  const rng = mulberry32(hash(`${types.join("/")}#${seed}#${k}`));
  const pick = (a) => a[Math.floor(rng() * a.length)];
  let s = pick(ROOTS[types[0]]);
  if (types[1]) s += pick(ROOTS[types[1]]);
  s += pick(ENDS);
  return s[0].toUpperCase() + s.slice(1);
}

// ---------- modelo ----------
async function fetchWithProgress(url, mb) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`no se pudo descargar ${url} (${r.status})`);
  const total = +r.headers.get("content-length") || mb * 2 ** 20, reader = r.body.getReader(), parts = [];
  let got = 0;
  $("bar").style.visibility = "visible";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    $("bar").firstElementChild.style.width = `${Math.min(100, (100 * got) / total)}%`;
    status(`Descargando el generador… ${(got / 2 ** 20).toFixed(1)} / ${mb} MB`);
  }
  $("bar").style.visibility = "hidden";
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  return buf;
}
async function getSession(key) {
  if (SESSIONS[key]) return SESSIONS[key];
  const m = META.models.find((x) => x.key === key);
  const bytes = await fetchWithProgress(m.file, m.mb);
  status("Preparando el generador…");
  let sess = null, ep = "WebAssembly";
  if (navigator.gpu && !autotest) {
    try {
      sess = await ort.InferenceSession.create(bytes.slice(), { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
      const out = await runRaw(sess, makeZ(1, 1, false), oneHot(["fire"]), 1);
      if (!out.every(Number.isFinite)) throw new Error("salida no finita");
      ep = "WebGPU";
    } catch (e) { console.warn("WebGPU no disponible; se usa WebAssembly.", e); sess = null; }
  }
  if (!sess) sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return (SESSIONS[key] = { sess, ep, meta: m });
}
async function runRaw(sess, z, y, n) {
  const res = await sess.run({ z: new ort.Tensor("float32", z, [n, META.z_dim]), y: new ort.Tensor("float32", y, [n, 18]) });
  return res.x.data;
}

// ---------- de tensor a sprite ----------
function viewFrom(data, k, v) {
  const rgb = new Uint8ClampedArray(PIX * 3), base = k * 6 * PIX + v * 3 * PIX;
  for (let p = 0; p < PIX; p++) for (let c = 0; c < 3; c++) rgb[p * 3 + c] = (data[base + c * PIX + p] + 1) * 127.5;
  // fondo: píxeles casi blancos conectados al borde (los blancos interiores, como ojos, se conservan)
  const bg = new Uint8Array(PIX), stack = [];
  const white = (p) => Math.min(rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]) >= WHITE;
  const seed = (p) => { if (!bg[p] && white(p)) { bg[p] = 1; stack.push(p); } };
  for (let i = 0; i < S; i++) { seed(i); seed((S - 1) * S + i); seed(i * S); seed(i * S + S - 1); }
  while (stack.length) {
    const p = stack.pop(), x = p % S, y = (p / S) | 0;
    if (x > 0) seed(p - 1); if (x < S - 1) seed(p + 1); if (y > 0) seed(p - S); if (y < S - 1) seed(p + S);
  }
  const mask = new Uint8Array(PIX);
  let x0 = S, x1 = -1, y0 = S, y1 = -1;
  for (let p = 0; p < PIX; p++) if (!bg[p]) {
    mask[p] = 1;
    const x = p % S, y = (p / S) | 0;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < 0) { x0 = 40; x1 = 56; y0 = 40; y1 = 56; }
  return { rgb, mask, bbox: { x0, x1, y0, y1 } };
}
function spriteCanvas(view, style) {
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const ctx = c.getContext("2d"), img = ctx.createImageData(S, S), d = img.data, { rgb, mask } = view;
  for (let p = 0; p < PIX; p++) {
    if (!mask[p]) continue;
    let r = rgb[p * 3], g = rgb[p * 3 + 1], b = rgb[p * 3 + 2];
    if (style === "retro") {
      const L = 0.299 * r + 0.587 * g + 0.114 * b;
      [r, g, b] = RETRO[L < 70 ? 0 : L < 130 ? 1 : L < 195 ? 2 : 3];
    }
    d[p * 4] = r; d[p * 4 + 1] = g; d[p * 4 + 2] = b; d[p * 4 + 3] = 255;
  }
  if (style === "contorno" || style === "retro") {
    const ink = style === "retro" ? RETRO[0] : [28, 28, 34];
    for (let p = 0; p < PIX; p++) {
      if (mask[p]) continue;
      const x = p % S, y = (p / S) | 0;
      if ((x > 0 && mask[p - 1]) || (x < S - 1 && mask[p + 1]) || (y > 0 && mask[p - S]) || (y < S - 1 && mask[p + S])) {
        d[p * 4] = ink[0]; d[p * 4 + 1] = ink[1]; d[p * 4 + 2] = ink[2]; d[p * 4 + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
// paleta dominante (5 colores) y coherencia frente/espalda: intersección de histogramas de 512 celdas (métrica del proyecto)
function palette(view, n = 5) {
  const cnt = new Map(), { rgb, mask } = view;
  for (let p = 0; p < PIX; p++) if (mask[p]) {
    const key = ((rgb[p * 3] >> 4) << 8) | ((rgb[p * 3 + 1] >> 4) << 4) | (rgb[p * 3 + 2] >> 4);
    const e = cnt.get(key) || [0, 0, 0, 0];
    e[0]++; e[1] += rgb[p * 3]; e[2] += rgb[p * 3 + 1]; e[3] += rgb[p * 3 + 2];
    cnt.set(key, e);
  }
  return [...cnt.values()].sort((a, b) => b[0] - a[0]).slice(0, n)
    .map((e) => `rgb(${Math.round(e[1] / e[0])},${Math.round(e[2] / e[0])},${Math.round(e[3] / e[0])})`);
}
function hist512(view) {
  const h = new Float64Array(512), { rgb, mask } = view;
  let tot = 0;
  for (let p = 0; p < PIX; p++) if (mask[p]) { h[((rgb[p * 3] >> 5) << 6) | ((rgb[p * 3 + 1] >> 5) << 3) | (rgb[p * 3 + 2] >> 5)]++; tot++; }
  if (tot) for (let i = 0; i < 512; i++) h[i] /= tot;
  return h;
}
function coherence(c) { const a = hist512(c.front), b = hist512(c.back); let s = 0; for (let i = 0; i < 512; i++) s += Math.min(a[i], b[i]); return s; }

// ---------- interfaz ----------
function status(msg) { $("status").textContent = msg; }
function busy(on) { for (const id of ["go", "all18", "surprise"]) $(id).disabled = on; }
function saveUI() { try { localStorage.setItem("pokegan-ui", JSON.stringify(ui)); } catch {} }
function setTypeVars(node, types) {
  node.style.setProperty("--t1", TYPE_COLORS[types[0]]);
  node.style.setProperty("--t2", TYPE_COLORS[types[1] || types[0]]);
}
function chips(types) { return types.map((t) => `<span class="chip" style="background:${TYPE_COLORS[t]}">${iconHTML(t)}${esName(t)}</span>`).join(""); }

// ---------- iconos de tipo y logotipo (pixel art) ----------
const ICON_CACHE = {};
function hexRGB(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function typeIconURL(t) {
  if (ICON_CACHE[t]) return ICON_CACHE[t];
  const c = document.createElement("canvas");
  c.width = c.height = 11;
  const g = c.getContext("2d"), [r, gg, b] = hexRGB(TYPE_COLORS[t]);
  const inside = (x, y) => (x - 5) ** 2 + (y - 5) ** 2 <= 31;
  for (let y = 0; y < 11; y++) for (let x = 0; x < 11; x++) {
    if (!inside(x, y)) continue;
    const rim = !inside(x - 1, y) || !inside(x + 1, y) || !inside(x, y - 1) || !inside(x, y + 1);
    const k = rim ? 0.62 : 1;
    g.fillStyle = `rgb(${(r * k) | 0},${(gg * k) | 0},${(b * k) | 0})`;
    g.fillRect(x, y, 1, 1);
  }
  const lum = (0.299 * r + 0.587 * gg + 0.114 * b) / 255;
  g.fillStyle = lum > 0.68 ? "#2b2b33" : "#ffffff";
  GLYPHS[t].forEach((row, y) => [...row].forEach((ch, x) => { if (ch === "#") g.fillRect(x + 1, y + 1, 1, 1); }));
  return (ICON_CACHE[t] = c.toDataURL());
}
function iconHTML(t) { return `<img class="ticon" src="${typeIconURL(t)}" alt="">`; }
function drawMascot() {
  const px = ["....tt....", "...XtXX...", "..XXXXXX..", ".XXoXXoXX.", ".XXXXXXXX.", ".XXXXXXXX.", "..XXXXXX..", "..XX..XX.."];
  const ctx = $("mascot").getContext("2d"), retro = document.body.dataset.theme === "retro";
  ctx.clearRect(0, 0, 10, 8);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 10; x++) {
    const ch = px[y][x];
    if (ch === ".") continue;
    ctx.fillStyle = retro ? (ch === "X" ? "#306230" : "#0f380f")
      : ch === "o" ? "#1d1d28" : ch === "t" ? TYPE_COLORS[st.types[1] || st.types[0]] : TYPE_COLORS[st.types[0]];
    ctx.fillRect(x, y, 1, 1);
  }
}
function drawWordmark() {
  const word = "POKEGAN", cv = $("wordmark"), W = word.length * 8 + 2, H = 11, retro = document.body.dataset.theme === "retro";
  cv.width = W; cv.height = H; cv.style.width = `${W * 4}px`;
  const g = cv.getContext("2d"), fill = new Uint8Array(W * H);
  [...word].forEach((ch, i) => FONT[ch].forEach((row, y) => [...row].forEach((v, x) => {
    if (v === "#") fill[(y + 1) * W + i * 8 + x + 1] = i < 4 ? 1 : 2;
  })));
  const at = (x, y) => (x >= 0 && y >= 0 && x < W && y < H ? fill[y * W + x] : 0);
  g.fillStyle = retro ? "#0f380f" : "#1d1d28";
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {   // contorno + sombra
    if (!at(x, y) && (at(x - 1, y) || at(x + 1, y) || at(x, y - 1) || at(x, y + 1) || at(x - 1, y - 1) || at(x, y - 2) || at(x - 1, y - 2))) g.fillRect(x, y, 1, 1);
  }
  const c1 = retro ? [139, 172, 15] : hexRGB(TYPE_COLORS[st.types[0]]);
  const c2 = retro ? [155, 188, 15] : hexRGB(TYPE_COLORS[st.types[1] || st.types[0]]);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = at(x, y);
    if (!v) continue;
    const base = v === 1 ? c1 : c2, light = y <= 2 ? 0.45 : y <= 4 ? 0.18 : 0;   // brillo arriba
    g.fillStyle = `rgb(${base.map((c) => Math.round(c + (255 - c) * light)).join(",")})`;
    g.fillRect(x, y, 1, 1);
  }
}
function seg(container, options, current, onPick) {
  container.innerHTML = "";
  for (const [val, label] of options) {
    const b = el("button", val === current ? "on" : "", label);
    b.type = "button";
    b.onclick = () => { for (const x of container.children) x.classList.remove("on"); b.classList.add("on"); onPick(val); };
    container.appendChild(b);
  }
}

function buildThemes() {
  const box = $("themes");
  box.innerHTML = "";
  for (const [key, label, color] of THEMES) {
    const b = el("button", key === ui.theme ? "on" : "", `<i style="background:${color || "linear-gradient(90deg,var(--t1),var(--t2))"}"></i>${label}`);
    b.type = "button";
    b.onclick = () => { ui.theme = key; saveUI(); applyTheme(); buildThemes(); };
    box.appendChild(b);
  }
}
function applyTheme() { document.body.dataset.theme = ui.theme; setTypeVars(document.body, st.types); drawMascot(); drawWordmark(); }

function typeChanged() { buildTypeSelects(); applyTheme(); }
function closeMenus(except) { document.querySelectorAll(".tpop").forEach((p) => { if (p !== except) p.hidden = true; }); }
function buildTypeSelects() {
  [0, 1].forEach((slot) => {
    const box = $(`tsel${slot}`), cur = st.types[slot] || null;
    box.innerHTML = "";
    const btn = el("button", "tbtn" + (cur ? "" : " none"),
      `${cur ? iconHTML(cur) : ""}<span class="nm">${cur ? esName(cur) : "Ninguno"}</span><svg><use href="#i-chev"/></svg>`);
    btn.type = "button";
    btn.setAttribute("aria-haspopup", "listbox");
    if (cur) btn.style.setProperty("--c", TYPE_COLORS[cur]);
    const pop = el("div", "tpop");
    pop.hidden = true;
    pop.setAttribute("role", "listbox");
    pop.onclick = (e) => e.stopPropagation();
    if (slot === 1) {
      const none = el("button", "topt none" + (cur ? "" : " on"), "Ninguno · un solo tipo");
      none.type = "button";
      none.onclick = () => { st.types = [st.types[0]]; typeChanged(); };
      pop.appendChild(none);
    }
    for (const [en, es] of META.types) {
      const o = el("button", "topt" + (en === cur ? " on" : ""), `${iconHTML(en)}${es}`);
      o.type = "button";
      o.style.setProperty("--c", TYPE_COLORS[en]);
      o.onclick = () => {
        if (slot === 0) st.types = [en, ...(st.types[1] && st.types[1] !== en ? [st.types[1]] : [])];
        else if (en !== st.types[0]) st.types = [st.types[0], en];
        typeChanged();
      };
      pop.appendChild(o);
    }
    btn.onclick = (e) => {
      e.stopPropagation();
      closeMenus(pop);
      pop.hidden = !pop.hidden;
      if (!pop.hidden) (pop.querySelector(".on") || pop.querySelector("button")).focus();
    };
    box.append(btn, pop);
  });
  $("unseen").hidden = !(st.types.length === 2 && UNSEEN.has(comboKey(st.types)));
}

function buildToolbar() {
  const bar = $("toolbar");
  bar.innerHTML = "";
  for (const [key, label, options] of STYLE_OPTS) {
    const g = el("div", "grp", `<span>${label}</span>`), s = el("div", "seg");
    seg(s, options, ui[key], (v) => { ui[key] = v; saveUI(); renderStage(); if (key === "sprite") renderGallery(); });
    g.appendChild(s);
    bar.appendChild(g);
  }
  const sw = el("label", "switch", `<input type="checkbox" ${ui.shadow === "on" ? "checked" : ""}>Sombra`);
  sw.querySelector("input").onchange = (e) => { ui.shadow = e.target.checked ? "on" : "off"; saveUI(); renderStage(); };
  bar.appendChild(sw);
}

// ---------- escenario ----------
function spot(view, extraClass) {
  const sp = el("div", "spot" + (extraClass ? " " + extraClass : ""));
  const { x0, x1, y1 } = view.bbox;
  sp.style.setProperty("--foot", `${((y1 + 1) / S) * 100}%`);
  sp.style.setProperty("--cx", `${((x0 + x1 + 1) / 2 / S) * 100}%`);
  sp.style.setProperty("--sw", `${Math.max(24, ((x1 - x0 + 1) / S) * 95)}%`);
  sp.appendChild(spriteCanvas(view, ui.sprite));
  return sp;
}
function renderStage() {
  clearInterval(flipTimer);
  const stage = $("stage"), c = st.creatures[st.sel];
  stage.className = `stage bg-${ui.bg} shadow-${ui.shadow}`;
  stage.style.setProperty("--scale", ui.scale);
  if (!c) return;
  setTypeVars(stage, c.types);
  stage.innerHTML = "";
  if (ui.layout === "ficha") {
    const row = el("div", "poses");
    for (const [view, label] of [[c.front, "FRENTE"], [c.back, "ESPALDA"]]) {
      const fig = el("figure", "pose");
      fig.append(spot(view), el("figcaption", "", label));
      row.appendChild(fig);
    }
    stage.appendChild(row);
  } else if (ui.layout === "combate") {
    const arena = el("div", "arena"), foe = el("div", "foe"), ally = el("div", "ally");
    const sf = spot(c.front), sb = spot(c.back);
    sf.prepend(el("div", "platform")); sb.prepend(el("div", "platform"));
    foe.append(sf, el("span", "vtag", "FRENTE"));
    ally.append(sb, el("span", "vtag", "ESPALDA"));
    arena.append(foe, ally);
    stage.appendChild(arena);
  } else {
    const wrap = el("div", "flipwrap"), flip = el("div", "flip"), f = el("div", "face front"), b = el("div", "face back");
    f.appendChild(spot(c.front)); b.appendChild(spot(c.back));
    flip.append(f, b);
    const tag = el("span", "vtag", "FRENTE");
    const turn = () => { flip.classList.toggle("flipped"); tag.textContent = flip.classList.contains("flipped") ? "ESPALDA" : "FRENTE"; };
    flip.onclick = () => { clearInterval(flipTimer); turn(); };
    wrap.append(flip, tag, el("span", "fliphint", "Haz clic para girar"));
    stage.appendChild(wrap);
    flipTimer = setInterval(turn, 2600);
  }
}
function renderInfo() {
  const c = st.creatures[st.sel], box = $("info");
  if (!c) { box.hidden = true; return; }
  box.hidden = false;
  setTypeVars(box, c.types);
  const coh = coherence(c);
  box.innerHTML = `
    <div><div class="num">N.º ${c.num} · SEMILLA ${c.seed} · ${MODEL_INFO[c.model][0].toUpperCase()}</div>
      <div class="name">${c.name}</div><div class="chips">${chips(c.types)}</div></div>
    <div><div class="pl">Paleta del frente</div><div class="sw">${palette(c.front).map((x) => `<i style="background:${x}"></i>`).join("")}</div>
      <div class="pl">Paleta de la espalda</div><div class="sw">${palette(c.back).map((x) => `<i style="background:${x}"></i>`).join("")}</div></div>
    <div><div class="pl">Coherencia de paleta frente/espalda</div><div class="meter"><i style="width:${Math.round(coh * 100)}%"></i></div>
      <div class="val">${coh.toFixed(2)} <span>· pares reales ≈ 0.63</span></div></div>
    <div class="dl"><button type="button" data-v="front" title="PNG con fondo transparente"><svg><use href="#i-down"/></svg>Frente</button>
      <button type="button" data-v="back" title="PNG con fondo transparente"><svg><use href="#i-down"/></svg>Espalda</button>
      <button type="button" data-v="ficha" title="Tarjeta con nombre, tipos y ambas vistas"><svg><use href="#i-down"/></svg>Ficha completa</button></div>`;
  box.querySelectorAll(".dl button").forEach((b) => (b.onclick = () => download(c, b.dataset.v)));
}
function renderGallery() {
  const g = $("gallery");
  g.innerHTML = "";
  st.creatures.forEach((c, k) => {
    const t = el("div", "thumb" + (k === st.sel ? " on" : "")), tp = el("div", "tp");
    tp.append(spriteCanvas(c.front, ui.sprite), spriteCanvas(c.back, ui.sprite));
    t.append(tp, el("div", "tn", `<span>${c.name}</span><span class="dots">${c.types.map((x) => iconHTML(x)).join("")}</span>`));
    t.title = `${c.name} · ${c.types.map(esName).join("/")} — frente y espalda`;
    t.onclick = () => { st.sel = k; renderAll(false); };
    g.appendChild(t);
  });
}
function renderAll(gallery = true) {
  renderStage(); renderInfo();
  if (gallery) renderGallery();
  else [...$("gallery").children].forEach((t, k) => t.classList.toggle("on", k === st.sel));
}

// ---------- descargas ----------
function saveCanvas(cv, name) { const a = document.createElement("a"); a.download = name; a.href = cv.toDataURL("image/png"); a.click(); }
function scaled(view, k) {
  const big = document.createElement("canvas");
  big.width = big.height = S * k;
  const g = big.getContext("2d");
  g.imageSmoothingEnabled = false;
  g.drawImage(spriteCanvas(view, ui.sprite), 0, 0, S * k, S * k);
  return big;
}
function download(c, what) {
  const base = `${c.name.toLowerCase()}_${c.types.join("-")}_s${c.seed}`;
  if (what !== "ficha") return saveCanvas(scaled(what === "front" ? c.front : c.back, 4), `${base}_${what === "front" ? "frente" : "espalda"}.png`);
  const W = 1000, H = 600, cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d"), c1 = TYPE_COLORS[c.types[0]], c2 = TYPE_COLORS[c.types[1] || c.types[0]];
  g.fillStyle = "#fffef9"; g.fillRect(0, 0, W, H);
  const band = g.createLinearGradient(0, 0, W, 0);
  band.addColorStop(0, c1); band.addColorStop(1, c2);
  g.fillStyle = band; g.fillRect(0, 0, W, 112);
  g.fillStyle = "#fff"; g.shadowColor = "rgba(0,0,0,.3)"; g.shadowBlur = 4;
  g.font = "700 46px Segoe UI, system-ui, sans-serif"; g.fillText(c.name, 40, 70);
  g.font = "500 18px Segoe UI, system-ui, sans-serif"; g.fillText(c.types.map(esName).join(" / "), 42, 98);
  g.shadowBlur = 0;
  g.font = "13px Consolas, monospace"; g.textAlign = "right"; g.fillText(`ESPÉCIMEN N.º ${c.num}`, W - 40, 40); g.textAlign = "left";
  const stage = g.createRadialGradient(W / 2, 300, 40, W / 2, 330, 520);
  stage.addColorStop(0, c1 + "33"); stage.addColorStop(1, c2 + "22");
  g.fillStyle = stage; g.fillRect(30, 132, W - 60, 400);
  [[c.front, "FRENTE", 150], [c.back, "ESPALDA", 562]].forEach(([v, label, x]) => {
    g.drawImage(scaled(v, 3), x, 150);
    g.fillStyle = "#1d2b24"; g.font = "600 13px Consolas, monospace"; g.textAlign = "center";
    g.fillText(label, x + 144, 456 + 40); g.textAlign = "left";
  });
  g.fillStyle = "#5d6b63"; g.font = "13px Consolas, monospace";
  g.fillText(`PokeGAN · modelo ${MODEL_INFO[c.model][0].toLowerCase()} · semilla ${c.seed} · coherencia de paleta ${coherence(c).toFixed(2)}`, 40, 572);
  saveCanvas(cv, `${base}_ficha.png`);
}

// ---------- generación ----------
function readSeed() { return Math.max(0, parseInt($("seed").value || "0", 10) || 0); }
async function produce(jobs, label) {
  const { sess, ep } = await getSession(st.model);
  const D = META.z_dim, out = [], t0 = performance.now();
  for (let s = 0; s < jobs.length; s += BATCH) {
    const part = jobs.slice(s, s + BATCH), m = part.length, z = new Float32Array(m * D), y = new Float32Array(m * 18);
    part.forEach((j, k) => { z.set(j.z, k * D); y.set(oneHot(j.types), k * 18); });
    status(`Generando ${s + 1}–${s + m} de ${jobs.length}…`);
    const data = await runRaw(sess, z, y, m);
    part.forEach((j, k) => out.push({
      front: viewFrom(data, k, 0), back: viewFrom(data, k, 1), types: j.types, seed: j.seed, model: st.model,
      name: creatureName(j.types, j.seed, j.k), num: String(1 + (hash(`${j.seed}:${j.k}`) % 9999)).padStart(4, "0"),
    }));
  }
  const ms = performance.now() - t0;
  status(`${label} en ${(ms / 1000).toFixed(2)} s · ${ep}`);
  return { out, ms };
}
async function generate() {
  busy(true);
  try {
    const seed = readSeed(), types = [...st.types], z = makeZ(st.n, seed, st.fid), D = META.z_dim;
    const jobs = Array.from({ length: st.n }, (_, k) => ({ z: z.subarray(k * D, (k + 1) * D), types, seed, k }));
    const { out, ms } = await produce(jobs, `${st.n} criaturas`);
    st.creatures = out; st.sel = 0; st.all18 = false;
    $("gtitle").textContent = `${types.map(esName).join(" / ")} · ${st.n} criaturas`;
    $("gmeta").textContent = `semilla ${seed}${st.fid ? " · modo fidelidad" : ""} · cada tarjeta: frente | espalda`;
    renderAll();
    return ms;
  } catch (e) { console.error(e); status("Error: " + e.message); throw e; }
  finally { busy(false); }
}
async function all18() {
  busy(true);
  try {
    const seed = readSeed(), z1 = makeZ(1, seed, st.fid);
    const jobs = META.types.map(([t], k) => ({ z: z1, types: [t], seed, k: 0 }));
    const { out } = await produce(jobs, "18 tipos");
    st.creatures = out; st.all18 = true;
    st.sel = Math.max(0, META.types.findIndex((p) => p[0] === st.types[0]));
    $("gtitle").textContent = "Misma criatura, 18 tipos";
    $("gmeta").textContent = `semilla ${seed} · el vector latente no cambia: solo cambia el tipo`;
    renderAll();
  } catch (e) { console.error(e); status("Error: " + e.message); }
  finally { busy(false); }
}

async function init() {
  META = await (await fetch("models/models.json")).json();
  applyTheme(); buildThemes(); buildTypeSelects(); buildToolbar();
  seg($("model"), META.models.map((m) => [m.key, MODEL_INFO[m.key]?.[0] || m.key]), st.model, (v) => {
    st.model = v; $("modelHint").textContent = MODEL_INFO[v]?.[1] || ""; st.all18 ? all18() : generate();
  });
  $("modelHint").textContent = MODEL_INFO[st.model][1];
  seg($("count"), [["1", "1"], ["4", "4"], ["6", "6"], ["8", "8"]], String(st.n), (v) => (st.n = +v));
  seg($("mode"), [["crudo", "Crudo"], ["fidelidad", "Fidelidad"]], "crudo", (v) => (st.fid = v === "fidelidad"));
  $("go").onclick = generate;
  $("all18").onclick = all18;
  $("surprise").onclick = () => {
    const keys = META.types.map((p) => p[0]), a = keys[Math.floor(Math.random() * 18)];
    const b = Math.random() < 0.55 ? keys[Math.floor(Math.random() * 18)] : null;
    st.types = b && b !== a ? [a, b] : [a];
    $("seed").value = Math.floor(Math.random() * 1e6);
    typeChanged();
    generate();
  };
  $("swap").onclick = () => { if (st.types.length === 2) { st.types = [st.types[1], st.types[0]]; typeChanged(); } };
  document.addEventListener("click", () => closeMenus(null));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenus(null); });
  $("seed").onkeydown = (e) => { if (e.key === "Enter") generate(); };
  try {
    const ms = await generate();
    if (autotest) document.title = `AUTOTEST ${$("gallery").children.length === st.n ? "OK" : "FALLA"} ${Math.round(ms)}ms`;
  } catch (e) { if (autotest) document.title = "AUTOTEST FALLA " + e.message; }
}

init();
