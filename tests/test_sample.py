"""Muestreo, CLI y figuras con un generador de juguete (CPU, sin checkpoint real)."""

import importlib.util
import sys
import types as pytypes
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from pokegan import types as T
from pokegan.sample import generate, interpolate, load_generator, make_y, make_z

ROOT = Path(__file__).resolve().parents[1]


class DummyG(nn.Module):
    """Generador de juguete: salida = tanh de una proyección fija de (z, y) a 6x12x12, ampliada a 96."""

    def __init__(self, z_dim: int = 120, bn: bool = False):
        super().__init__()
        self.z_dim = z_dim
        g = torch.Generator().manual_seed(0)
        self.wz = nn.Parameter(torch.randn(z_dim, 6 * 12 * 12, generator=g) / z_dim ** 0.5)
        self.wy = nn.Parameter(torch.randn(18, 6 * 12 * 12, generator=g))
        self.bn = nn.BatchNorm2d(6) if bn else None

    def forward(self, z, y, lam: float = 1.0):
        h = (z @ self.wz + lam * y @ self.wy).view(-1, 6, 12, 12)
        if self.bn is not None:
            h = self.bn(h)
        return torch.tanh(F.interpolate(h, scale_factor=8, mode="nearest"))


@pytest.fixture
def fake_models(monkeypatch):
    """Registra DummyG como pokegan.models.generator.Generator y un standing_stats que registra llamadas."""
    calls = []
    gen_mod = pytypes.ModuleType("pokegan.models.generator")
    gen_mod.Generator = DummyG
    ema_mod = pytypes.ModuleType("pokegan.ema")
    ema_mod.standing_stats = lambda G, ys, n_batches=32, batch=64, lam=1.0: calls.append((ys, lam))
    monkeypatch.setitem(sys.modules, "pokegan.models.generator", gen_mod)
    monkeypatch.setitem(sys.modules, "pokegan.ema", ema_mod)
    return calls


def _fake_ckpt(path: Path, bn: bool = False) -> Path:
    g, g_ema = DummyG(bn=bn), DummyG(bn=bn)
    with torch.no_grad():
        g_ema.wz.mul_(2.0)  # G y G_ema distintos para saber cuál se cargó
    bank = torch.from_numpy(np.stack([T.encode(["fire"]), T.encode(["water", "flying"])] * 8))
    torch.save({
        "G": g.state_dict(), "G_ema": g_ema.state_dict(), "step": 1234, "lam": 0.75,
        "cfg": {"G": {"z_dim": 120, "bn": bn}}, "label_bank": bank.to(torch.uint8),
    }, path)
    return path


def test_load_generator_ema_and_info(tmp_path, fake_models):
    ckpt = _fake_ckpt(tmp_path / "last.pt")
    G, info = load_generator(ckpt, ema=True, device="cpu", standing=False)
    assert not G.training and info["step"] == 1234 and info["lam"] == 0.75
    assert info["cfg"]["G"]["z_dim"] == 120 and tuple(info["label_bank"].shape) == (16, 18)
    G_raw, _ = load_generator(ckpt, ema=False, device="cpu", standing=False)
    assert torch.allclose(G.wz, 2 * G_raw.wz)
    assert fake_models == []  # standing=False no recalcula BN
    load_generator(ckpt, device="cpu", standing=True)
    (ys, lam), = fake_models
    assert lam == 0.75 and ys.dtype == torch.float32 and tuple(ys.shape) == (16, 18)


def test_load_generator_real_standing_stats(tmp_path, monkeypatch):
    try:
        import pokegan.ema  # noqa: F401
    except Exception as e:  # el módulo lo escribe otra tarea
        pytest.skip(f"pokegan.ema no disponible: {e}")
    gen_mod = pytypes.ModuleType("pokegan.models.generator")
    gen_mod.Generator = DummyG
    monkeypatch.setitem(sys.modules, "pokegan.models.generator", gen_mod)
    G, _ = load_generator(_fake_ckpt(tmp_path / "bn.pt", bn=True), device="cpu", standing=True)
    assert not G.training
    assert G.bn.running_mean.abs().sum() > 0  # standing stats cambió las medias de BN
    a = generate(G, ["fire"], 2, seed=3)
    G2, _ = load_generator(tmp_path / "bn.pt", device="cpu", standing=True)
    assert (generate(G2, ["fire"], 2, seed=3) == a).all()  # reproducible entre cargas


@pytest.mark.parametrize("types", [["fuego"], ["Hada"], ["fire"], ["fire", "fairy"], "fuego/volador"])
def test_generate_accepts_es_en_and_unseen(types):
    out = generate(DummyG(), types, 3, seed=0)
    assert out.shape == (3, 2, 96, 96, 3) and out.dtype == np.uint8


def test_generate_seed_determinism_and_conditioning():
    G = DummyG()
    a = generate(G, ["fire", "fairy"], 4, seed=7)
    assert (a == generate(G, ["fire", "fairy"], 4, seed=7)).all()
    assert not (a == generate(G, ["fire", "fairy"], 4, seed=8)).all()
    assert not (a == generate(G, ["water"], 4, seed=7)).all()  # el tipo cambia la salida
    assert not (a[0] == a[1]).all()  # z distintos dentro del lote


def test_generate_random_conditions_and_truncation():
    G = DummyG()
    bank = torch.from_numpy(np.stack([T.encode(["ice", "poison"])] * 5))
    y = make_y(None, 6, seed=1, label_bank=bank)
    assert (y == bank[0]).all()
    assert make_y(None, 6, seed=1).sum(1).min() >= 1
    out = generate(G, None, 5, seed=1, label_bank=bank)
    assert out.shape == (5, 2, 96, 96, 3)
    assert (out == generate(G, None, 5, seed=1, label_bank=bank)).all()
    z = make_z(64, 120, seed=0, truncation=0.5)
    assert z.abs().max() <= 0.5 and (z == make_z(64, 120, seed=0, truncation=0.5)).all()
    with pytest.raises(ValueError):
        generate(G, ["pikachu"], 1)


def test_interpolate_endpoints():
    G = DummyG()
    zi = interpolate(G, ["fire"], ["fire"], 0, 1, steps=5)
    assert zi.shape == (5, 2, 96, 96, 3)
    assert (zi[0] == generate(G, ["fire"], 1, seed=0)[0]).all()
    assert (zi[-1] == generate(G, ["fire"], 1, seed=1)[0]).all()
    ti = interpolate(G, ["fuego"], ["agua"], 3, 3, steps=4)
    assert (ti[0] == generate(G, ["fire"], 1, seed=3)[0]).all()
    assert (ti[-1] == generate(G, ["water"], 1, seed=3)[0]).all()


def _png_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0 and Image.open(p).size[0] > 0


def test_figures_write_files(tmp_path):
    from pokegan import figures as Fg

    G = DummyG()
    assert _png_ok(Fg.grid_by_type(G, [("fire",), ("dragon", "fairy")], 2, 0, tmp_path / "g.png"))
    assert _png_ok(Fg.grid_by_type(G, [("water",)], 2, 0, tmp_path / "gp.png", post=True))
    assert _png_ok(Fg.same_z_all_types(G, 1, 0, tmp_path / "s.png", scale=1))
    assert _png_ok(Fg.interpolation_strip(G, ["fire"], ["water"], 0, 0, tmp_path / "i.png", steps=4))
    gif = Fg.interpolation_gif(G, ["fire"], ["fire"], 0, 1, tmp_path / "i.gif", steps=4, scale=1)
    assert _png_ok(gif) and Image.open(gif).n_frames == 6  # 4 + vuelta (2)
    assert _png_ok(Fg.raw_vs_post(G, [("grass", "poison")], 0, tmp_path / "r.png", n_z=1))
    assert _png_ok(Fg.save_pairs_grid(generate(G, None, 3, 0), tmp_path / "p.png",
                                      labels=["a", "b", "c"]))
    bank = np.stack([T.encode(["fire", "flying"])] * 3 + [T.encode(["water", "ground"])])
    assert Fg.frequent_duals(bank, 2) == [("flying", "fire"), ("ground", "water")]
    log = tmp_path / "log.csv"
    rows = ["step,time_min,it_s,lam,d_loss,g_loss,d_real,d_fake,d_mis,r_t,r_v,div,lecam,"
            "kid_front,kid_back,type_map,coherence"]  # columnas de train.LOG_COLS + EVAL_COLS
    for s in range(0, 5000, 100):
        kid = "0.05,0.06,0.4,0.5" if s % 1000 == 0 else ",,,"
        r_v = "" if s < 1000 else "0.3"
        rows.append(f"{s},{s / 100},3.5,{min(1, s / 3000):.2f},1.5,0.8,0.6,-0.4,-0.6,0.5,{r_v},0.2,0.01,{kid}")
    log.write_text("\n".join(rows) + "\n")
    assert _png_ok(Fg.training_curves(log, tmp_path / "c.png"))
    log.write_text("step,lam,d_loss,g_loss,d_real,d_fake,d_mis,r_t,div\n0,0,1,1,0,0,0,0,1\n")
    assert _png_ok(Fg.training_curves(log, tmp_path / "c2.png"))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_cli(tmp_path, fake_models, capsys):
    ckpt = _fake_ckpt(tmp_path / "last.pt")
    cli = _load_script("generate")
    out = tmp_path / "gen"
    cli.main(["--ckpt", str(ckpt), "--types", "fuego", "hada", "--n", "3", "--post",
              "--out", str(out), "--device", "cpu"])
    run = next(out.iterdir())
    assert _png_ok(run / "grilla.png") and _png_ok(run / "grilla_post.png")
    assert Image.open(run / "00_frente.png").size == (96, 96)
    assert Image.open(run / "02_espalda.png").size == (96, 96)
    assert Image.open(run / "00_par_x4.png").size == (4 * 192, 4 * 96)
    assert _png_ok(run / "01_frente_post.png")
    assert str(run) in capsys.readouterr().out
    cli.main(["--ckpt", str(ckpt), "--n", "2", "--out", str(out), "--device", "cpu", "--no-ema"])
    with pytest.raises(SystemExit):
        cli.main(["--ckpt", str(ckpt), "--types", "pikachu", "--out", str(out), "--device", "cpu"])


def test_make_figures_cli(tmp_path, fake_models):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = _fake_ckpt(run_dir / "last.pt")
    (run_dir / "log.csv").write_text(
        "step,lam,d_loss,g_loss,d_real,d_fake,d_mis,r_t,div\n0,0,1,1,0,0,0,0,1\n100,0.1,1,1,0,0,0,0,1\n")
    out = tmp_path / "figs"
    _load_script("make_figures").main(["--ckpt", str(ckpt), "--out", str(out), "--n-z", "2",
                                        "--device", "cpu", "--scale", "1"])
    names = {p.name for p in out.iterdir()}
    assert {"01_monotipos.png", "03_combos_no_vistos.png", "05_interp_z.gif",
            "06_interp_tipo_fuego_agua.gif", "08_curvas_entrenamiento.png"} <= names
    assert all(p.stat().st_size > 0 for p in out.iterdir())
