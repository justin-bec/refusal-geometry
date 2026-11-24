#!/usr/bin/env python
"""
10_gains.py -- how much does ADDING a direction move the margin of the refusal, per unit of norm?
Article: "Adding and removing, measured: (H1)". Llama-3-8B-Instruct by default. Published prompts
(07_thresholds.py); nothing is generated or optimized.

OBJECTS.
  x = y + u m       template state (layer c), u = <x, m>; margin delta = u - u*(y); refusal iff delta > 0.
  gain of d         adding t d (d unit) changes u by t <d, m> EXACTLY and changes u* only through the part of d
                    orthogonal to m, d_perp = d - <d, m> m. So the margin gain per unit of norm is
                    G(d) = <d, m> - (u*(y + t d_perp) - u*(y - t d_perp)) / (2 t),
                    and it suffices to MEASURE the threshold after adding +t d_perp and -t d_perp (dichotomy along m,
                    in the context of the prompt). For d = m, G = 1 exactly (nothing to measure).
  directions d      e1^concat; the direction of Arditi et al. copied on the P template positions; a random
                    direction; all unit vectors of R^{P d} oriented so that <d, m> >= 0.
  prompts           --n_prompts published prompts with a measured threshold u*_0, round robin over groups.
  dichotomy         window [u*_0 - H, u*_0 + H] (H = --half_width) around the prompt's own threshold, --rounds rounds
                    of --k responses; the threshold without addition is remeasured in the same pass.

RULES: all measures of a round (prompts x directions x signs x amplitudes) in the same batches, texts sorted by
length; the state is injected by a hook at prefill only; no repeated message; the script keeps its own journal;
every round is saved, the same command resumes.
DURATION (defaults): 40 prompts x (3 directions x 2 signs x 2 amplitudes + 1) x 5 rounds x 6 responses = 15 600
responses, about 4 h.

USAGE: python experiments/10_gains.py   (after 07_thresholds.py)
Writes runs/10_gains/<model>/: journal.log, plan.pt, dichotomy.pt, results.json, figures/gains.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli, dichotomy  # noqa: E402
from refusal.data import stratified  # noqa: E402
from refusal.hooks import Inject  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import figures_written, phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.plots import mpl  # noqa: E402
from refusal.stats import bootstrap_ci, unit  # noqa: E402

COLOURS = {"m": "#000000", "e1^concat": "#CC79A7", "Arditi et al. (P positions)": "#D55E00",
           "random direction": "#999999"}


def parse(argv=None):
    ap = cli.parser("Margin gain per unit of norm of added directions.", models=("llama3",))
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--t", default="1.5,3", help="amplitudes (norm of the addition)")
    ap.add_argument("--half_width", type=float, default=2.5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--max_prompt_tokens", type=int, default=1500)
    return ap.parse_args(argv)


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    T = cli.load("07_thresholds", key, "thresholds.pt")
    Dd, A = cli.directions(key, args.directions), cli.arditi(key)
    m = unit(T["m"].double().numpy())
    P = Dd["P"]
    ts = [float(x) for x in args.t.split(",")]
    dirs = {"e1^concat": unit(Dd["e1_concat"].double().numpy()),
            "Arditi et al. (P positions)": unit(np.tile(A["direction"].double().numpy(), P)),
            "random direction": unit(np.random.default_rng(args.seed + 1).standard_normal(m.size))}
    dirs = {k: (v if v @ m >= 0 else -v) for k, v in dirs.items()}
    ok = [i for i in range(len(T["texts"])) if np.isfinite(T["ustar"][i])]
    chosen = stratified(ok, lambda i: T["groups"][i], args.n_prompts, args.seed)
    meas = [(i, None, 0, 0.0) for i in chosen] + [(i, d, s, t) for i in chosen for d in dirs for s in (1, -1)
                                                  for t in ts]
    X = T["X"].double().numpy()
    states = np.stack([X[i] if d is None else X[i] + s * t * (dirs[d] - (dirs[d] @ m) * m) for i, d, s, t in meas])
    clock(f"{len(chosen)} prompts, {len(meas)} measures, {len(meas) * args.rounds * args.k} responses")

    def _dicho():
        eng = Engine(get_spec(key), args.dtype, args.device, args.mem_frac, args.seed,
                     max_prompt_tokens=args.max_prompt_tokens)
        inj = eng.add(Inject(eng.model, eng.spec.layer, eng.template))
        ids = {i: eng.encode(T["texts"][i]) for i in chosen}
        order = list(ids)
        keys = [{"ctx": order.index(i), "state": j, "lo": float(T["ustar"][i]) - args.half_width,
                 "hi": float(T["ustar"][i]) + args.half_width} for j, (i, _, _, _) in enumerate(meas)]
        r = dichotomy.run(eng, judge, inj, keys, [ids[i] for i in order], [T["texts"][i] for i in order],
                          torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(m, dtype=torch.float32),
                          args.rounds, args.k, S / "dichotomy_state.pt", clock, args.tokens, args.batch,
                          args.batch_judge)
        eng.free()
        return {"meas": meas, "threshold": r["threshold"], "censored": r["censored"]}
    Z = phase("dichotomy", S / "dichotomy.pt", redo, clock, _dicho)
    thr = {ms: t for ms, t in zip(Z["meas"], Z["threshold"])}
    res = {"m": {"G": 1.0, "note": "exact: adding t m changes u by t and leaves u* unchanged"}}
    print("\n" + "=" * 104)
    print("MARGIN GAIN PER UNIT OF NORM, G(d) = <d, m> - slope of the threshold (95 % CI over prompts)")
    for d in dirs:
        res[d] = {"cos_with_m": float(dirs[d] @ m)}
        for t in ts:
            sl = np.array([(thr[(i, d, 1, t)] - thr[(i, d, -1, t)]) / (2 * t) for i in chosen])
            sl = sl[np.isfinite(sl)]
            ci = bootstrap_ci(sl)
            res[d][str(t)] = {"G": float(dirs[d] @ m - ci[0]), "G_ci": [float(dirs[d] @ m - ci[2]),
                                                                      float(dirs[d] @ m - ci[1])],
                              "threshold_slope": ci[0], "n": int(len(sl))}
            r = res[d][str(t)]
            print(f"  {d:30s} t = {t:<4g} G = {r['G']:+.3f} [{r['G_ci'][0]:+.3f}; {r['G_ci'][1]:+.3f}]   "
                  f"(<d, m> = {res[d]['cos_with_m']:+.3f}, threshold slope {r['threshold_slope']:+.3f}, {r['n']} prompts "
                  f"without censored threshold)")
    print(f"  {'m':30s} G = 1 exactly")
    print("=" * 104)
    json.dump(res, open(S / "results.json", "w"), indent=1)
    plt = mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["m"] + list(dirs)
    xs = np.arange(len(names))
    w = 0.8 / len(ts)
    for j, t in enumerate(ts):
        g = [1.0 if n == "m" else res[n][str(t)]["G"] for n in names]
        lo = [0 if n == "m" else g[q] - res[n][str(t)]["G_ci"][0] for q, n in enumerate(names)]
        hi = [0 if n == "m" else res[n][str(t)]["G_ci"][1] - g[q] for q, n in enumerate(names)]
        ax.bar(xs + (j - (len(ts) - 1) / 2) * w, g, w, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], capsize=3,
               color=[COLOURS[n] for n in names], alpha=0.45 + 0.55 * j / max(1, len(ts) - 1),
               label=f"amplitude t = {t:g} (darker = larger)")
    ax.axhline(1, color="black", ls="--", lw=0.8, label="gain of m (exactly 1)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set(ylabel="change of the margin u - u* per unit of norm added",
           title=f"Margin gain of added directions, {len(chosen)} published prompts (dichotomy along m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(S / "figures" / "gains.png", dpi=130)
    plt.close(fig)
    figures_written([S / "figures" / "gains.png"])


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    judge = None
    for key in cli.model_keys(args):
        judge = judge or make_judge(args.judge, dtype_of(args.dtype), args.device)
        run_model(key, args, judge, root)


if __name__ == "__main__":
    main()
