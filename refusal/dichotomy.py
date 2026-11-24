"""Thresholds along an axis m, by dichotomy, for many (state, context) keys at once.

Coordinates: x = y + u m, u = <x, m>. For a key (state s, context c), the template state at the main layer of the
prompt c is replaced by the state of s with its coordinate along m imposed, x' = x_s + (u - <x_s, m>) m, and u* is
the value of u where the refusal crosses 1/2. Refusal at a point = mean J over its coherent responses. Each round
halves every interval; all keys and all their responses go through the same batches. After each round the state
is saved, and the same command resumes. A threshold stuck at a bound of its interval is CENSORED.
"""
import numpy as np
import torch

from .judge import coherent_mean_J


def run(engine, judge, inject, keys, prompts, texts, states, m, rounds, k, path, clock, tokens=64, batch=32,
        batch_judge=24):
    """keys: list of dicts {"ctx": prompt index, "state": state index, "lo": float, "hi": float}; prompts: id
    tensors; texts: prompt texts; states: (N, P*d) float tensor; m: (P*d,) unit axis; ``inject``: an Inject hook
    on the main layer. Returns {"lo", "hi", "history", "censored", "threshold"} (numpy arrays per key)."""
    P, d = engine.P, engine.d
    m = torch.as_tensor(m, dtype=torch.float32)
    if path.exists():
        Z = torch.load(path, weights_only=False)
        if len(Z["lo"]) != len(keys):
            raise RuntimeError(f"{path.name} was computed for other keys: move it away or use --redo")
    else:
        Z = {"round": 0, "lo": np.array([kk["lo"] for kk in keys], float),
             "hi": np.array([kk["hi"] for kk in keys], float), "history": [[] for _ in keys]}
    lo0 = np.array([kk["lo"] for kk in keys], float)
    hi0 = np.array([kk["hi"] for kk in keys], float)
    while Z["round"] < rounds:
        mid = 0.5 * (Z["lo"] + Z["hi"])
        rows = [(j, s) for j in range(len(keys)) for s in range(k)]

        def setup(r, am):
            st = []
            for x in r:
                j = rows[x][0]
                y = states[keys[j]["state"]].float()
                st.append((y + (float(mid[j]) - float(y @ m)) * m).reshape(P, d))
            inject.set(torch.stack(st), torch.ones(len(r), dtype=torch.bool))

        tag = f"round {Z['round'] + 1}/{rounds}"
        reps = engine.sample([prompts[keys[j]["ctx"]] for j, _ in rows], setup, 1.0, tokens, batch, clock,
                             f"dichotomy {tag}", path.with_name(f"_partial_{path.stem}_gen.pt"))
        v = judge([(texts[keys[j]["ctx"]], engine.decode(reps[x][0])) for x, (j, _) in enumerate(rows)], batch_judge,
                  clock, checkpoint=path.with_name(f"_partial_{path.stem}_judge.pt"))
        for j in range(len(keys)):
            sel = range(j * k, (j + 1) * k)
            q = coherent_mean_J([v[x] for x in sel], [reps[x][0] for x in sel])
            Z["history"][j].append((float(mid[j]), q))
            if np.isfinite(q) and q >= 0.5:
                Z["hi"][j] = mid[j]
            else:
                Z["lo"][j] = mid[j]
        Z["round"] += 1
        torch.save(Z, path)
        clock(f"  dichotomy: {tag} done")
    cens = (Z["lo"] <= lo0 + 1e-9) | (Z["hi"] >= hi0 - 1e-9)
    thr = 0.5 * (Z["lo"] + Z["hi"])
    return {"lo": Z["lo"], "hi": Z["hi"], "history": Z["history"], "censored": cens,
            "threshold": np.where(cens, np.nan, thr)}
