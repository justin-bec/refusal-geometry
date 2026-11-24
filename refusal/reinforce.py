"""Gradient of the judged refusal with respect to the template state, by REINFORCE.

q(x) = E_{s ~ p(.|x)} [J(s)], the expectation over responses sampled at temperature 1 when the template state is x.
For any constant b independent of s, grad q(x) = E[(J(s) - b) grad log p(s|x)]. With k responses, the constant of
response s is the mean J of the k - 1 others (leave one out): independent of s, so the estimator is unbiased.
Only the model is differentiated (through the layers above the state); the judge does not depend on x.

Each gradient is estimated ``n_est`` times on independent sets of k responses (g^1, g^2, ...): cross products
<g^1_n, g^2_k> are then unbiased estimates of <g_n, g_k>, including n = k.
"""
import numpy as np
import torch

from .generation import align, run_batches
from .log import phase


def normals(engine, judge, items, texts, setup, capture, k, n_est, out_dir, tag, clock, redo=(), tokens=64,
            batch_gen=32, batch_grad=8, batch_judge=24):
    """items: list of prompt id tensors (one per point where the gradient is wanted); texts: their prompt texts
    (for the judge); setup(item_indices, am): sets, for each row, the intervention that places the template state at
    the point (transplantation dose, injected state...). ``capture``: a GradCapture registered after those hooks.
    Saves three phases in out_dir (responses, judgments, gradients) and returns
    {"G": (n_est, N, P, d), "q": (n_est, N) mean J of each set}."""
    K = n_est * k
    rows = [(i, s) for i in range(len(items)) for s in range(K)]

    def _responses():
        reps = engine.sample([items[i] for i, _ in rows], lambda r, am: setup([rows[x][0] for x in r], am),
                             1.0, tokens, batch_gen, clock, f"{tag}: responses")
        return {"rows": rows, "reps": reps}

    R = phase(f"{tag}_responses", out_dir / f"{tag}_a_responses.pt", redo, clock, _responses)
    if R["rows"] != rows:
        raise RuntimeError(f"{tag}_a_responses.pt was computed for other items or another k: use --redo")
    reps = R["reps"]

    def _judgments():
        v = judge([(texts[i], engine.decode(reps[x][0])) for x, (i, _) in enumerate(rows)], batch_judge, clock)
        return {"J": [d["J"] for d in v]}

    J = np.array(phase(f"{tag}_judgments", out_dir / f"{tag}_b_judgments.pt", redo, clock, _judgments)["J"])
    Jr = J.reshape(len(items), n_est, k)
    tot = Jr.sum(-1, keepdims=True)
    weights = ((Jr - (tot - Jr) / (k - 1)) / k).reshape(-1)       # leave-one-out baseline, within each set

    def _gradients():
        P, d = engine.P, engine.d
        G = torch.zeros(n_est, len(items), P, d)
        dev = engine.dev

        def one(xs):
            pr = [list(items[rows[x][0]]) for x in xs]
            rp = [reps[x][1] for x in xs]
            ids, am, pos, n_max, mask = align(pr, rp, engine.pad_id)
            tail = torch.tensor([len(r) for r in rp], dtype=torch.long)
            try:
                setup([rows[x][0] for x in xs], am)
                capture.active = True
                with torch.enable_grad(), engine.template.batch(tail, ids):
                    lg = engine.model(input_ids=ids.to(dev), attention_mask=am.to(dev), position_ids=pos.to(dev),
                                      use_cache=False, logits_to_keep=n_max + 1).logits[:, :n_max].float()
                    lp = torch.log_softmax(lg, -1)
                    tl = lp.gather(-1, ids[:, ids.shape[1] - n_max:].to(dev)[..., None]).squeeze(-1)
                    w = torch.tensor([weights[x] for x in xs], device=dev, dtype=torch.float32)
                    ((tl * mask.to(dev)).sum(1) * w).sum().backward()
                g = capture.grad()
            finally:
                capture.clear()
                engine.clear()
            return [g[j] for j in range(len(xs))]

        out = run_batches(len(rows), lambda x: len(items[rows[x][0]]) + len(reps[x][1]), batch_grad, one, clock,
                          f"{tag}: backpropagation")
        for x, (i, s) in enumerate(rows):
            G[s // k, i] += out[x]
        return {"G": G, "q": Jr.mean(-1).T}

    res = phase(f"{tag}_gradients", out_dir / f"{tag}_c_gradients.pt", redo, clock, _gradients)
    G = res["G"].reshape(n_est, len(items), -1)
    if n_est >= 2:
        cos = (G[0] * G[1]).sum(1) / (G[0].norm(dim=1) * G[1].norm(dim=1)).clamp(min=1e-30)
        clock(f"  {tag}: cos between the two estimates of a normal, median {float(cos.median()):.3f}, "
              f"negative for {int((cos < 0).sum())}/{len(cos)} ; mean refusal at the points "
              f"{float(res['q'].mean()):.3f} (aimed at 0.5)")
    return res
