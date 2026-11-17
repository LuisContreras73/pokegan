import torch

from pokegan.losses import (
    LeCam, d_hinge, derangement_other_species, g_hinge, mode_seeking, transition_lambda,
)


def t(v):
    return torch.tensor([float(v)] * 4)


def test_hinge_values_and_signs():
    assert d_hinge(t(2), t(-2)).item() == 0.0          # D seguro y correcto ⇒ sin pérdida
    assert d_hinge(t(0), t(0)).item() == 2.0
    assert d_hinge(t(0), t(0), mis=t(0)).item() == 2.5  # ½·relu(1+0)
    assert d_hinge(t(-1), t(1)).item() == 4.0           # D invertido ⇒ pérdida grande
    assert g_hinge(t(-1)).item() == 1.0                 # G quiere C(fake) alto
    assert d_hinge(t(0), t(0), mis=torch.empty(0)).item() == 2.0
    # máscara: solo cuentan los desajustados válidos (aquí el único válido tiene logit 1 ⇒ relu(2)=2)
    m = torch.tensor([5.0, 1.0, 5.0, 5.0])
    ok = torch.tensor([False, True, False, False])
    assert d_hinge(t(2), t(-2), mis=m, mis_mask=ok).item() == 1.0  # 0.5 · 2


def test_pairwise_mismatch_has_no_drift_in_any_regime():
    """Término por pares + real/falso a peso 1: D constante no siente tirón ni al inicio ni en el régimen
    tardío (desajustados ya hundidos en −2). Con GAN-CLS absoluto el régimen tardío da −½ (deriva a +1)."""
    for mis_val in (None, -2.0):  # None: desajustado = c (inicio); -2: D ya separa la coherencia
        c = torch.tensor(0.3, requires_grad=True)
        v = c.expand(4)
        m = v if mis_val is None else torch.full((4,), mis_val)
        d_hinge(v, v, mis=m, mis_weight=0.5, fake_weight=1.0, mis_pair_real=v).backward()
        assert abs(c.grad.item()) < 1e-6
    c = torch.tensor(0.3, requires_grad=True)
    v = c.expand(4)
    d_hinge(v, v, mis=torch.full((4,), -2.0), mis_weight=0.5, fake_weight=0.5).backward()
    assert abs(c.grad.item() - (-0.5)) < 1e-6  # GAN-CLS tardío: el descenso sube c hacia +1


def test_balanced_weights_constant_D_has_no_pull_toward_minus_one():
    """Pesos GAN-CLS (real 1, falso ½, desajustado ½): con D constante c ∈ (−1, 1) el gradiente es 0.

    Con pesos desbalanceados (real 1, falso 1, desajustado ½) el gradiente empuja c hacia −1, donde las
    bisagras negativas se apagan: esa fue la causa del colapso de D (diagnóstico del 2026-11-16).
    """
    for fw, expected in ((0.5, 0.0), (1.0, 0.5)):
        c = torch.tensor(0.3, requires_grad=True)
        v = c.expand(4)
        d_hinge(v, v, mis=v, mis_weight=0.5, fake_weight=fw).backward()
        assert abs(c.grad.item() - expected) < 1e-6  # >0 ⇒ el descenso baja c hacia −1


def test_transition_lambda():
    assert transition_lambda(0, 100, 300) == 0.0
    assert transition_lambda(100, 100, 300) == 0.0
    assert transition_lambda(200, 100, 300) == 0.5
    assert transition_lambda(1000, 100, 300) == 1.0
    assert transition_lambda(5, 10, 10) == 0.0 and transition_lambda(10, 10, 10) == 1.0


def test_lecam_penalty_zero_at_anchors_and_state():
    lc = LeCam(0.9)
    assert lc.penalty(t(1), t(-1)).item() == 0.0  # sin anclas aún
    lc.update(t(1), t(-1))
    assert lc.penalty(t(-1), t(1)).item() == 0.0   # real ≤ α_F y fake ≥ α_R ⇒ 0
    assert lc.penalty(t(3), t(-3)).item() == 32.0  # (3−(−1))² + (1−(−3))²
    lc2 = LeCam()
    lc2.load_state_dict(lc.state_dict())
    assert lc2.ema_real == 1.0 and lc2.ema_fake == -1.0
    assert lc2.penalty(t(3), t(-3)).item() == 32.0     # funciona con anclas float tras reanudar


def test_mode_seeking_penalizes_identical_outputs():
    z1, z2 = torch.zeros(2, 4), torch.ones(2, 4)
    same = mode_seeking(torch.ones(2, 3), torch.ones(2, 3), z1, z2)
    diff = mode_seeking(torch.ones(2, 3), -torch.ones(2, 3), z1, z2)
    assert same > diff


def test_derangement_no_self_and_mask_other_species():
    torch.manual_seed(0)
    sp = torch.tensor([1, 2, 3, 4, 5, 6, 7, 7])
    for _ in range(20):
        idx, valid = derangement_other_species(sp)
        assert (idx != torch.arange(8)).all()
        assert sorted(idx.tolist()) == list(range(8))
        assert torch.equal(valid, sp[idx] != sp)


def test_alpha_shape_penalty_area_and_tv():
    from pokegan.losses import alpha_shape_penalty

    empty = torch.zeros(1, 2, 96, 96)
    blob = torch.zeros(1, 2, 96, 96)
    blob[..., 30:60, 30:60] = 1                     # 900 px compactos
    frag = torch.zeros(1, 2, 96, 96)
    for k in range(9):                              # 9 fragmentos de 10x10 = 900 px
        r, c = divmod(k, 3)
        frag[..., 10 + 30 * r:20 + 30 * r, 10 + 30 * c:20 + 30 * c] = 1
    pa_e, _ = alpha_shape_penalty(empty, tv_target=0.0)
    pa_b, tv_b = alpha_shape_penalty(blob, tv_target=0.0)
    _, tv_f = alpha_shape_penalty(frag, tv_target=0.0)
    assert pa_e.item() == 1.0 and pa_b.item() == 0.0   # vacío penalizado, cuerpo real no
    assert tv_f > 2.5 * tv_b                           # misma área, fragmentado ⇒ más perímetro (TV)
    # área visible desde la imagen compuesta: un lienzo blanco (α>0 pero RGB blanco) cuenta como vacío
    white = torch.ones(1, 6, 96, 96)
    pa_w, _ = alpha_shape_penalty(blob, x=white)
    assert pa_w.item() > 0.98  # = 1 − (9216·σ(−7.85))/240 ≈ 0.985
    _, p_tv = alpha_shape_penalty(blob)               # TV del bloque compacto < objetivo real ⇒ sin penalización
    assert p_tv.item() == 0.0
