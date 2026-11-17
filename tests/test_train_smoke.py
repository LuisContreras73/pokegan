import csv
from pathlib import Path

import pytest
import torch
import yaml

from pokegan.ema import EMA, standing_stats
from pokegan.models import Generator
from pokegan.train import LOG_COLS, train

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "pairs96.npz"


def test_ema_copies_then_averages_and_standing_stats():
    torch.manual_seed(0)
    G = Generator(ch=8, attn=False)
    ema = EMA(G, decay=0.5, start=2)
    with torch.no_grad():
        for p in G.parameters():
            p.add_(1.0)
    ema.update(G, step=1)  # antes de start ⇒ copia exacta
    assert all(torch.equal(a, b) for a, b in zip(ema.module.parameters(), G.parameters()))
    before = [p.clone() for p in ema.module.parameters()]
    with torch.no_grad():
        for p in G.parameters():
            p.add_(2.0)
    ema.update(G, step=5)  # promedio con d = 0.5 ⇒ +1
    assert all(torch.allclose(a, b + 1.0) for a, b in zip(ema.module.parameters(), before))
    bn = next(m for m in ema.module.modules() if isinstance(m, torch.nn.BatchNorm2d))
    old = bn.running_mean.clone()
    ys = torch.zeros(10, 18)
    ys[:, 3] = 1
    standing_stats(ema.module, ys, n_batches=2, batch=4)
    assert not torch.equal(old, bn.running_mean) and not ema.module.training


@pytest.mark.skipif(not CACHE.exists(), reason="ejecuta scripts/prepare_data.py primero")
def test_smoke_train_and_resume(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    cfg["out_dir"] = str(tmp_path)
    cfg["data"]["cache"] = str(CACHE)
    run = train(cfg)
    assert (run / "last.pt").exists()
    assert sorted(p.name for p in (run / "samples").glob("*.png"))[:1] == ["step_000002.png"]
    rows = list(csv.DictReader(open(run / "log.csv", encoding="utf-8")))
    assert set(LOG_COLS) <= set(rows[0]) and len(rows) == 4
    ck = torch.load(run / "last.pt", weights_only=False)
    assert ck["step"] == 4 and ck["label_bank"].shape[1] == 18 and "G_ema" in ck
    assert ck["sched"] == {"t_start": 1, "t_end": 2} and set(ck["lecam"]) == {"uc", "cc"}

    with pytest.raises(SystemExit):  # no pisar una corrida existente sin --resume
        train(dict(cfg))

    cfg["resume"] = str(run / "last.pt")
    cfg["train"]["iters"] = 8
    train(cfg)
    ck = torch.load(run / "last.pt", weights_only=False)
    assert ck["step"] == 8 and ck["lam"] == 1.0
    rows = list(csv.DictReader(open(run / "log.csv", encoding="utf-8")))
    assert [int(r["step"]) for r in rows] == list(range(1, 9))
