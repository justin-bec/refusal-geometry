"""Checks of the interventions on a tiny random model (run after tests/make_tiny.py, with REFUSAL_MODELS set):
    python -m pytest tests/test_hooks.py -q
"""
import torch

from refusal.generation import first_real, pad_left
from refusal.hooks import Ablation, GradCapture, Inject, StateReader, Transplant
from refusal.models import Engine, get_spec

E = Engine(get_spec("llama3"), dtype="float32", device="cpu", mem_frac=None)
IDS = [E.encode(t) for t in ("how to bake bread?", "write a short poem about the sea please, now")]


def read_after(hook_setup, layer=3):
    reader = StateReader(E.model, [layer], E.template, stop=True)
    ids, am, pos = pad_left(IDS, E.pad_id)
    with E.template.batch(torch.zeros(len(IDS), dtype=torch.long), ids):
        hook_setup(am)
        S = reader.read(E.model, ids, am, pos)[layer]
    E.clear()
    reader.remove()
    return S


def test_diagonal_ablation_and_offset():
    u = torch.randn(E.d)
    u = u / u.norm()
    abl = E.add(Ablation(E.model, E.template))
    U = torch.zeros(2, E.L, E.d)
    U[:, 3] = u
    S = read_after(lambda am: abl.set(U=U, start=first_real(am)))
    assert float((S @ u).abs().max()) < 1e-4
    off = torch.zeros(2, E.L)
    off[:, 3] = 0.7
    S = read_after(lambda am: abl.set(U=U, offset=off, start=first_real(am)))
    assert float(((S @ u) - 0.7).abs().max()) < 1e-4
    flags = torch.tensor([[True, False, True]] * 2)            # template spared
    S0 = read_after(lambda am: None)
    S = read_after(lambda am: abl.set(U=U, flags=flags, start=first_real(am)))
    tpl = slice(None)                                          # S holds the template positions only: untouched
    assert float(((S[:, tpl] - S0[:, tpl]) @ u).abs().max()) < 1e-4
    abl.remove()
    E.hooks.remove(abl)


def test_coupled_ablation():
    v = torch.randn(E.P, E.d)
    v = v / v.norm()
    abl = E.add(Ablation(E.model, E.template))
    V = torch.zeros(2, E.L, E.P, E.d)
    V[:, 3] = v
    S = read_after(lambda am: abl.set(V=V, start=first_real(am)))
    assert float((S.reshape(2, -1) @ v.reshape(-1)).abs().max()) < 1e-4
    abl.remove()
    E.hooks.remove(abl)


def test_inject_and_transplant():
    inj = E.add(Inject(E.model, 3, E.template))
    X = torch.randn(2, E.P, E.d)
    S = read_after(lambda am: inj.set(X, torch.tensor([True, False])))
    S0 = read_after(lambda am: None)
    assert torch.allclose(S[0], X[0], atol=1e-5) and torch.allclose(S[1], S0[1], atol=1e-5)
    inj.remove()
    E.hooks.remove(inj)
    tr = E.add(Transplant(E.model, 3, E.template))
    mu = torch.randn(E.L, E.P, E.d)
    S = read_after(lambda am: tr.set(torch.tensor([1.0, 0.0]), mu))
    assert torch.allclose(S[0], mu[3], atol=1e-5) and torch.allclose(S[1], S0[1], atol=1e-5)
    tr.remove()
    E.hooks.remove(tr)


def test_grad_capture_sees_injection():
    inj = E.add(Inject(E.model, 3, E.template))
    cap = E.add(GradCapture(E.model, 3, E.template))
    X = torch.randn(2, E.P, E.d)
    ids, am, pos = pad_left(IDS, E.pad_id)
    with E.template.batch(torch.zeros(2, dtype=torch.long), ids):
        inj.set(X, torch.tensor([True, True]))
        cap.active = True
        out = E.model(input_ids=ids, attention_mask=am, position_ids=pos).logits[:, -1].sum()
        out.backward()
        g = cap.grad()
    E.clear()
    assert g.shape == (2, E.P, E.d) and float(g.abs().sum()) > 0
    for h in (inj, cap):
        h.remove()
        E.hooks.remove(h)
