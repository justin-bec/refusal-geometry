#!/usr/bin/env python
"""
07_thresholds.py -- the refusal THRESHOLD of published prompts along the axis m, prompt by prompt, and for each
(base request, attack) pair, the threshold of each state placed in the context of the other.
Article: "An axis: (H1)", "A threshold per prompt" (data; analysis in 11_threshold_analysis.py). Llama-3-8B-Instruct
by default. Published prompts only (jailbreak artifacts, transformations, benchmarks, harmless prompts that look
suspicious, and their base requests): nothing is generated or optimized.

OBJECTS (layer c, template, m of 02_directions.py).
  x_n      natural template state of prompt n at the input of layer c (P positions); u_n = <x_n, m>.
  q_n      natural refusal: mean J of --k_nat responses without intervention; refused if q_n > 1/2.
  u*(s, c) threshold of the state of prompt s placed in the CONTEXT of prompt c (its tokens): the template state of
           c is replaced by x_s + (u - <x_s, m>) m and u* is the value of u where the refusal crosses 1/2, found by
           dichotomy on [--u_min, --u_max] (--rounds rounds of --k responses; precision (u_max - u_min) / 2^rounds).
           A threshold stuck at a bound is CENSORED.
  keys     u*(n, n) for every prompt; for every pair (b base, a attack): u*(b, a) (state of the base, context of the
           attack) and u*(a, b) (state of the attack, context of the base).
  remeasures  --remeasures independent remeasures (other random draws, --rounds_fine rounds of --k_fine responses) of
           the four thresholds of every pair, u*(b, b), u*(a, a), u*(b, a), u*(a, b): cross products between
           independent measures remove the noise shared by the decomposition (11_threshold_analysis.py).
PROMPTS: per group (a dataset x a method or transformation; refusal.data.published_groups), --n_pairs pairs if the
group has base requests, else --n_alone prompts; deduplicated. Groups the model cannot read are excluded (--exclude).

RULES: all keys of a dichotomy round and all their responses in the same batches, texts sorted by length; the
state is injected by a hook at prefill only; no repeated message; the script keeps its own journal; every phase and
every round is saved (partial saves inside long rounds): the same command resumes.
DURATION (defaults, ~1 300 prompts, ~550 pairs): first measure ~2 400 keys x 5 rounds x 6 responses + natural
refusal ~ 80 000 responses (about 20 h at 0.9 s per judged response); each remeasure ~2 200 keys x 6 x 8 ~ 105 000.
Start with --remeasures 0 and add them later: the same command reuses what is done.

USAGE: python experiments/07_thresholds.py [--remeasures 2]   (after 02_directions.py and data/download.py)
Writes runs/07_thresholds/<model>/: journal.log, prompts.json, natural.pt, states.pt, measure_0.pt, measure_<r>.pt,
thresholds.pt.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli, dichotomy  # noqa: E402
from refusal.data import published_groups  # noqa: E402
from refusal.hooks import Inject  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402

EXCLUDE = "sorry_morse,sorry_caesar,sorry_atbash,sorry_ascii,jbb_JBC"     # encodings the model cannot read


def parse(argv=None):
    ap = cli.parser("Thresholds along m of published prompts, and of each state in the context of its pair.",
                    models=("llama3",))
    ap.add_argument("--n_pairs", type=int, default=16, help="pairs per attack group")
    ap.add_argument("--n_alone", type=int, default=16, help="prompts per group without base requests")
    ap.add_argument("--exclude", default=EXCLUDE)
    ap.add_argument("--k_nat", type=int, default=8)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--remeasures", type=int, default=0)
    ap.add_argument("--k_fine", type=int, default=8)
    ap.add_argument("--rounds_fine", type=int, default=6)
    ap.add_argument("--u_min", type=float, default=-3.0)
    ap.add_argument("--u_max", type=float, default=5.0)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--max_prompt_tokens", type=int, default=1500)
    return ap.parse_args(argv)


def choose(args, clock):
    G = published_groups(log=clock)
    excl = [e for e in args.exclude.split(",") if e]
    rng = random.Random(args.seed)
    prompts, index, pairs = [], {}, []

    def add(text, group):
        if text not in index:
            index[text] = len(prompts)
            prompts.append({"text": text, "groups": [group]})
        elif group not in prompts[index[text]]["groups"]:
            prompts[index[text]]["groups"].append(group)
        return index[text]

    for g in sorted(G):
        if any(g.startswith(e) for e in excl):
            continue
        el = list(dict.fromkeys(G[g]))
        rng.shuffle(el)
        with_base = [(p, b) for p, b in el if b]
        if with_base:
            for p, b in with_base[:args.n_pairs]:
                ia, ib = add(p, g), add(b, "base_of_" + g)
                if ia != ib:
                    pairs.append({"group": g, "base": ib, "attack": ia})
        else:
            for p, _ in el[:args.n_alone]:
                add(p, g)
    return prompts, pairs


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    D = cli.directions(key, args.directions)
    m = D["m"].float().reshape(-1)
    fp = S / "prompts.json"
    if fp.exists() and "prompts" not in redo:
        Pj = json.load(open(fp))
    else:
        pr, pa = choose(args, clock)
        Pj = {"prompts": pr, "pairs": pa}
        json.dump(Pj, open(fp, "w"), indent=1, ensure_ascii=False)
    prompts, pairs = Pj["prompts"], Pj["pairs"]
    texts = [p["text"] for p in prompts]
    groups = [p["groups"][0] for p in prompts]
    clock(f"{len(prompts)} prompts in {len(set(groups))} groups, {len(pairs)} (base, attack) pairs")
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed, max_prompt_tokens=args.max_prompt_tokens)
    c = spec.layer
    ids = [eng.encode(t) for t in texts]

    def _natural():
        rows = [(i, s) for i in range(len(ids)) for s in range(args.k_nat)]
        reps = eng.sample([ids[i] for i, _ in rows], None, 1.0, args.tokens, args.batch, clock, "natural refusal",
                          S / "_partial_natural_gen.pt")
        v = judge([(texts[i], eng.decode(reps[x][0])) for x, (i, _) in enumerate(rows)], args.batch_judge, clock,
                  checkpoint=S / "_partial_natural_judge.pt")
        J = np.array([x["J"] for x in v]).reshape(len(ids), args.k_nat)
        H = np.array([x["harmful"] is True for x in v], float).reshape(len(ids), args.k_nat)
        return {"q": J.mean(1), "harmful": H.mean(1)}
    Nat = phase("natural", S / "natural.pt", redo, clock, _natural)
    X = phase("states", S / "states.pt", redo, clock, lambda: {"X": eng.template_states(
        ids, [c], max(1, args.batch_tf // 2), clock, "states")[c].reshape(len(ids), -1)})["X"]
    u = (X @ m).numpy()
    clock(f"natural refusal: {int((Nat['q'] > 0.5).sum())}/{len(ids)} prompts refused; u from {u.min():+.2f} to "
          f"{u.max():+.2f}")

    inj = eng.add(Inject(eng.model, c, eng.template))
    keys0 = [(i, i) for i in range(len(ids))] + [(p["base"], p["attack"]) for p in pairs] + \
            [(p["attack"], p["base"]) for p in pairs]
    keys0 = list(dict.fromkeys(keys0))

    def measure(keys, rounds, k, r):
        if f"measure_{r}" in redo and (S / f"measure_{r}_state.pt").exists():
            (S / f"measure_{r}_state.pt").unlink()
        eng.gen.manual_seed(args.seed + 1000 * r)
        kd = [{"state": s, "ctx": cx, "lo": args.u_min, "hi": args.u_max} for s, cx in keys]
        res = dichotomy.run(eng, judge, inj, kd, ids, texts, X, m, rounds, k, S / f"measure_{r}_state.pt", clock,
                            args.tokens, args.batch, args.batch_judge)
        return {"keys": keys, "threshold": res["threshold"], "censored": res["censored"], "lo": res["lo"],
                "hi": res["hi"], "history": res["history"]}

    M = [phase("measure_0", S / "measure_0.pt", redo, clock, lambda: measure(keys0, args.rounds, args.k, 0))]
    kf = list(dict.fromkeys([kk for p in pairs for kk in ((p["base"], p["base"]), (p["attack"], p["attack"]),
                                                          (p["base"], p["attack"]), (p["attack"], p["base"]))]))
    for r in range(1, args.remeasures + 1):
        M.append(phase(f"measure_{r}", S / f"measure_{r}.pt", redo, clock,
                       lambda r=r: measure(kf, args.rounds_fine, args.k_fine, r)))
    eng.free()
    own = {kk: t for kk, t in zip(M[0]["keys"], M[0]["threshold"])}
    ustar = np.array([own[(i, i)] for i in range(len(ids))])
    torch.save({"texts": texts, "groups": groups, "pairs": pairs, "q": Nat["q"], "harmful": Nat["harmful"], "X": X,
                "u": u, "m": m, "ustar": ustar, "measures": M, "u_bounds": (args.u_min, args.u_max), "layer": c},
               S / "thresholds.pt")
    clock(f"thresholds: {int(np.isfinite(ustar).sum())}/{len(ids)} measured, "
          f"{int((~np.isfinite(ustar)).sum())} censored; {len(M) - 1} remeasures of the pairs; "
          f"written to {S / 'thresholds.pt'} (analysis: 11_threshold_analysis.py)")


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    judge = None
    for key in cli.model_keys(args):
        judge = judge or make_judge(args.judge, dtype_of(args.dtype), args.device)
        run_model(key, args, judge, root)


if __name__ == "__main__":
    main()
