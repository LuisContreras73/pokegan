"""Demo interactiva de PokeGAN: elige modelo, tipo(s) y semilla → frente y espalda al instante.

Uso (desde c:\\Pokemon):
    python scripts/demo_app.py            # abre http://127.0.0.1:8000 en el navegador
    python scripts/demo_app.py --port 8010 --no-browser

Los modelos se cargan una sola vez (G_ema + standing stats) y quedan en memoria; cada generación tarda < 1 s en GPU.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from PIL import Image  # noqa: E402

from pokegan import types as T  # noqa: E402
from pokegan.postprocess import clean_pair  # noqa: E402
from pokegan.refine import truncate  # noqa: E402
from pokegan.sample import load_generator, make_y, make_z, render  # noqa: E402

MODELS = {
    "vision": ("Vision (discriminador auxiliar ConvNeXt-T)", ROOT / "weights" / "pokegan_vision.pt"),
    "pura": ("Pura (todo desde cero)", ROOT / "weights" / "pokegan_pura.pt"),
}
_CACHE: dict[str, tuple] = {}
_LOCK = threading.Lock()
app = FastAPI(title="PokeGAN")


def get_model(key: str):
    if key not in MODELS or not MODELS[key][1].exists():
        raise HTTPException(404, f"modelo no disponible: {key}")
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = load_generator(MODELS[key][1], ema=True)
    return _CACHE[key]


def png_b64(img: np.ndarray, scale: int = 4) -> str:
    im = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/api/models")
def models():
    return [{"key": k, "label": v[0]} for k, v in MODELS.items() if v[1].exists()]


@app.get("/api/generate")
def api_generate(model: str = "pura", t1: str = "fire", t2: str = "", seed: int = 0, n: int = 4,
                 modo: str = "crudo", post: bool = False):
    G, info = get_model(model)
    types = [t1] + ([t2] if t2 and t2 != t1 else [])
    try:
        T.encode(types)
    except ValueError as e:
        raise HTTPException(400, str(e))
    n = max(1, min(n, 12))
    z = make_z(n, G.z_dim, seed)
    if modo == "fidelidad":  # truncamiento solo en z0–z2 (silueta), ver docs/matematica.md §10b
        z = truncate(z, 1.0, chunks=(0, 1, 2), generator=torch.Generator().manual_seed(seed))
    y = make_y(types, n, seed)
    pairs = render(G, z, y, lam=info["lam"])  # uint8 [n, 2, 96, 96, 3]
    out = []
    for k, p in enumerate(pairs):
        f, b = (clean_pair(p[0], p[1], colors=10, keep_frac=0.15) if post else (p[0], p[1]))
        out.append({"k": k, "frente": png_b64(f), "espalda": png_b64(b)})
    return JSONResponse({"condicion": T.label(T.encode(types)), "modelo": MODELS[model][0], "paso": info["step"],
                         "semilla": seed, "modo": modo, "post": post, "criaturas": out})


PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8"><title>PokeGAN — demo</title>
<style>
body{font-family:Segoe UI,system-ui,sans-serif;background:#f4f3ee;color:#1d2b24;margin:0}
.wrap{max-width:1250px;margin:0 auto;padding:28px;display:grid;grid-template-columns:300px 1fr;gap:28px}
h1{font-size:38px;line-height:1.05;margin:6px 0 20px} .kicker{font:12px monospace;letter-spacing:.08em;color:#3c6b4f}
label{display:block;font-weight:600;font-size:14px;margin:14px 0 6px} select,input{width:100%;padding:10px;border:1px solid #c9cfc7;border-radius:8px;font-size:15px;background:#fff;box-sizing:border-box}
.row{display:flex;gap:8px} .row input{flex:1} button{padding:12px 14px;border:0;border-radius:8px;font-size:15px;cursor:pointer}
#go{width:100%;margin-top:18px;background:#22402f;color:#fff;font-weight:600} #rnd{background:#fff;border:1px solid #c9cfc7}
.chk{display:flex;align-items:center;gap:8px;margin-top:10px;font-size:14px} .chk input{width:auto}
.panel{background:#fffef9;border:1px solid #dcdcd2;border-radius:16px;padding:20px}
.head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px} .head h2{margin:0;font-size:26px}
.meta{font:12px monospace;color:#555} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}
.card{border:1px solid #e3e3da;border-radius:12px;padding:10px} .pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.view{background:#fff radial-gradient(#e6e6e6 1px,transparent 1px);background-size:12px 12px;border:1px solid #ececec;border-radius:10px;text-align:center}
.view img{width:100%;image-rendering:pixelated} .cap{display:flex;justify-content:space-between;font-size:13px;padding:4px 2px}
.cap a{font:11px monospace;color:#22402f} .note{font-size:12px;color:#666;margin-top:14px;line-height:1.4}
</style></head><body><div class="wrap"><div>
<div class="kicker">GENERADOR CONDICIONAL · POKEGAN</div><h1>Crea una criatura.</h1>
<label>Modelo</label><select id="model"></select>
<label>Tipo principal</label><select id="t1"></select>
<label>Segundo tipo</label><select id="t2"></select>
<label>Semilla</label><div class="row"><input id="seed" type="number" value="0"><button id="rnd" title="semilla al azar">🎲</button></div>
<label>Cantidad</label><select id="n"><option>1</option><option>2</option><option selected>4</option><option>6</option><option>8</option></select>
<label>Muestreo</label><select id="modo"><option value="crudo">Crudo (métricas oficiales)</option><option value="fidelidad">Modo fidelidad (truncamiento z0–z2)</option></select>
<div class="chk"><input type="checkbox" id="post"><span>Post-proceso pixel art (declarado)</span></div>
<button id="go">Generar criaturas ↗</button>
<div class="note">Un mismo z genera el <b>frente</b> y la <b>espalda</b> de la misma criatura; la única condición es el tipo.
Misma semilla ⇒ mismas criaturas (para comparar modelos o tipos).</div></div>
<div class="panel"><div class="head"><h2 id="title">—</h2><span class="meta" id="meta"></span></div><div class="grid" id="grid"></div></div></div>
<script>
const TYPES = __TYPES__;
const t1=document.getElementById('t1'), t2=document.getElementById('t2');
t2.add(new Option('Ninguno',''));
for (const [en,es] of TYPES){ t1.add(new Option(es,en)); t2.add(new Option(es,en)); }
t1.value='fire';
fetch('/api/models').then(r=>r.json()).then(ms=>{const s=document.getElementById('model'); for(const m of ms) s.add(new Option(m.label,m.key));});
document.getElementById('rnd').onclick=()=>{document.getElementById('seed').value=Math.floor(Math.random()*1e9); gen();};
document.getElementById('go').onclick=gen;
async function gen(){
  const q=new URLSearchParams({model:model.value,t1:t1.value,t2:t2.value,seed:seed.value,n:document.getElementById('n').value,
    modo:modo.value,post:document.getElementById('post').checked});
  document.getElementById('title').textContent='Generando…';
  const r=await fetch('/api/generate?'+q); if(!r.ok){document.getElementById('title').textContent='Error: '+(await r.text()); return;}
  const d=await r.json();
  document.getElementById('title').textContent=d.condicion;
  document.getElementById('meta').textContent=d.modelo+' · paso '+d.paso+' · semilla '+d.semilla+(d.modo==='fidelidad'?' · modo fidelidad':'')+(d.post?' · post-proceso':'');
  const g=document.getElementById('grid'); g.innerHTML='';
  for(const c of d.criaturas){
    const f='data:image/png;base64,'+c.frente, b='data:image/png;base64,'+c.espalda, nm=d.condicion.replace('/','-')+'_s'+d.semilla+'_'+c.k;
    g.insertAdjacentHTML('beforeend',`<div class="card"><div class="pair">
      <div><div class="view"><img src="${f}"></div><div class="cap"><span>Frente</span><a download="${nm}_frente.png" href="${f}">Descargar PNG ↓</a></div></div>
      <div><div class="view"><img src="${b}"></div><div class="cap"><span>Espalda</span><a download="${nm}_espalda.png" href="${b}">Descargar PNG ↓</a></div></div>
      </div></div>`);
  }
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    import json

    return PAGE.replace("__TYPES__", json.dumps([[t, T.TYPES_ES[t]] for t in T.TYPES], ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    import uvicorn

    url = f"http://127.0.0.1:{a.port}"
    print(f"PokeGAN demo en {url}  (Ctrl+C para salir)", flush=True)
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
