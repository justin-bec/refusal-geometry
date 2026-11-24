#!/usr/bin/env python
"""
09_h0.py -- (H0) along m: is the region a threshold region along m, and does the refusal increase along m?
Article: "(H0) along m". Llama-3-8B-Instruct by default. Published prompts (07_thresholds.py); nothing is generated
or optimized.

TWO PARTS.
  LOCAL (CPU, seconds; anchors of 08_slopes.py). A threshold region with a differentiable threshold is monotone
  along a direction r if and only if r makes no obtuse angle with any inner normal of the border. At each anchor the
  normal g is estimated twice independently (g^1, g^2); for each direction d: both <g^1, d>, <g^2, d> > 0 (acute
  angle), both < 0 (CONFIRMED violation: two independent estimates agree), opposite signs (undecided, noise);
  cosine cos(g, d) = <(g^1 + g^2)/2, d> / sqrt(<g^1, g^2>) (<g^1, g^2> estimates |g|^2 without bias).
  Directions (unit vectors of R^{P d}): m; e1^concat; the direction of Arditi et al. copied on the P template
  positions; a random direction. Selection bias: anchors are points where the dichotomy along m found a threshold.
  FINE (GPU). For --n_prompts prompts (round robin over groups, threshold not censored), the template state is
  replaced by X_n + t m with t = (u*_n - u_n) + offset, for the offsets --offsets around the threshold of each
  prompt, --k responses per point. Refusal of a response: J >= 1/2 and coherent. Measures:
    descent share   rho = sum of the downward variations / total variation of the curves (Jordan decomposition),
                    pooled over curves; 0 for increasing curves. Compared with its value under noise alone: binary
                    responses drawn from each curve's isotonic (increasing) fit, same numbers of responses;
    largest drop    per curve, max over t < t' of q_t - q_t', with a parametric bootstrap p-value (Benjamini-
                    Hochberg over curves).

RULES (fine part): all prompts and offsets mixed in the same batches, texts sorted by length; the state is injected
by a hook at prefill only; no repeated message; the script keeps its own journal; one phase = one file.

USAGE: python experiments/09_h0.py [--part local|fine|both]   (after 08_slopes.py for local, 07 for fine)
Writes runs/09_h0/<model>/: journal.log, local.json, fine.pt, fine.json, figures/h0_local.png, figures/h0_fine.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.data import stratified  # noqa: E402
from refusal.generation import trigram_diversity  # noqa: E402
from refusal.hooks import Inject  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import figures_written, phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.plots import mpl  # noqa: E402
from refusal.stats import benjamini_hochberg, isotonic, monotonicity_test, unit  # noqa: E402

COLOURS = {"m": "#000000", "e1^concat": "#CC79A7", "Arditi et al. (P positions)": "#D55E00",
           "random direction": "#999999"}


def parse(argv=None):
    ap = cli.parser("(H0) along m: local angle condition at the anchors, and refusal curves around the threshold.",
                    models=("llama3",))
    ap.add_argument("--part", default="both", choices=["local", "fine", "both"])
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--offsets", default="-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n_sim", type=int, default=500, help="simulations of the noise-only curves")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--max_prompt_tokens", type=int, default=1500)
    return ap.parse_args(argv)


def directions(key, m, P, seed, source):
    D, A = cli.directions(key, source), cli.arditi(key)
    e1c = unit(D["e1_concat"].double().numpy())
    ar = unit(np.tile(A["direction"].double().numpy(), P))
    rnd = unit(np.random.default_rng(seed + 1).standard_normal(m.size))
    out = {"m": m, "e1^concat": e1c, "Arditi et al. (P positions)": ar, "random direction": rnd}
    return {k: (v if v @ m >= 0 else -v) for k, v in out.items()}


def local(key, S, args):
    An = cli.load("08_slopes", key, "anchors.pt")
    G1, G2 = An["G1"].double().numpy(), An["G2"].double().numpy()
    m = unit(An["m"].double().numpy())
    P = cli.directions(key, args.directions)["P"]
    dirs = directions(key, m, P, args.seed, args.directions)
    n2 = (G1 * G2).sum(1)
    ok = n2 > 0
    res, cosines = {}, {}
    print("\n" + "=" * 104)
    print(f"(H0) LOCAL: angle between the normal at each of the {len(G1)} anchors and each direction")
    print(f"  {'direction':30s} {'cos(d, m)':>9s} {'acute':>8s} {'obtuse (violation)':>19s} {'undecided':>10s} "
          f"{'median cos':>11s}")
    for k, d in dirs.items():
        a1, a2 = G1 @ d, G2 @ d
        pos, neg = (a1 > 0) & (a2 > 0), (a1 < 0) & (a2 < 0)
        c = (0.5 * (a1 + a2))[ok] / np.sqrt(n2[ok])
        cosines[k] = c
        res[k] = {"cos_with_m": float(d @ m), "acute": float(pos.mean()), "obtuse": float(neg.mean()),
                  "undecided": float(1 - pos.mean() - neg.mean()), "median_cos": float(np.median(c)),
                  "obtuse_count": int(neg.sum()), "n": int(len(G1))}
        r = res[k]
        print(f"  {k:30s} {r['cos_with_m']:+9.3f} {r['acute'] * 100:7.1f}% {r['obtuse'] * 100:12.1f}% "
              f"({r['obtuse_count']:3d}) {r['undecided'] * 100:9.1f}% {r['median_cos']:+11.3f}")
    print("  obtuse: both independent estimates make an obtuse angle with d -- a confirmed local violation of the")
    print("  monotonicity along d")
    print("=" * 104)
    json.dump(res, open(S / "local.json", "w"), indent=1)
    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    ax = axs[0]
    for k in dirs:
        ax.hist(cosines[k], bins=np.linspace(-1, 1, 81), histtype="step", lw=1.8, color=COLOURS[k], density=True,
                label=f"{k} (median {res[k]['median_cos']:+.2f})")
    ax.axvline(0, color="#D55E00", ls="--", lw=1, label="0: on the left, monotonicity along d is violated")
    ax.set(xlabel="cosine between the normal to the border (anchor) and the direction d", ylabel="density",
           title=f"Normals at the {len(G1)} anchors: angle with each direction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[1]
    xs = np.arange(len(dirs))
    bottom = np.zeros(len(dirs))
    for key_, lab, col in (("acute", "acute angle (both estimates)", "#009E73"),
                           ("undecided", "undecided (estimates of opposite signs)", "#CCCCCC"),
                           ("obtuse", "obtuse angle in both estimates: local violation", "#D55E00")):
        v = np.array([res[k][key_] for k in dirs])
        ax.bar(xs, v, bottom=bottom, color=col, label=lab)
        bottom += v
    ax.set_xticks(xs)
    ax.set_xticklabels(list(dirs), fontsize=9)
    ax.set(ylim=(0, 1.02), ylabel="share of the anchors", title="Does the direction make an acute angle with every normal?")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Local condition of monotonicity, from normals estimated twice independently", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(S / "figures" / "h0_local.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return S / "figures" / "h0_local.png"


def descent(curves):
    """Pooled descent share: sum of downward variations / total variation."""
    dn = tot = 0.0
    for q in curves:
        dq = np.diff(q)
        dn += np.clip(-dq, 0, None).sum()
        tot += np.abs(dq).sum()
    return dn / max(tot, 1e-12)


def fine(key, S, args, judge, clock, redo):
    T = cli.load("07_thresholds", key, "thresholds.pt")
    ok = [i for i in range(len(T["texts"])) if np.isfinite(T["ustar"][i])]
    sel = stratified(ok, lambda i: T["groups"][i], args.n_prompts, args.seed)
    offs = [float(x) for x in args.offsets.split(",")]

    def _fine():
        eng = Engine(get_spec(key), args.dtype, args.device, args.mem_frac, args.seed,
                     max_prompt_tokens=args.max_prompt_tokens)
        inj = eng.add(Inject(eng.model, eng.spec.layer, eng.template))
        m, X = T["m"].float(), T["X"].float()
        ids = {i: eng.encode(T["texts"][i]) for i in sel}
        rows = [(i, o, s) for i in sel for o in offs for s in range(args.k)]

        def setup(r, am):
            st = [(X[rows[x][0]] + float(T["ustar"][rows[x][0]] - T["u"][rows[x][0]] + rows[x][1]) * m)
                  .reshape(eng.P, -1) for x in r]
            inj.set(torch.stack(st), torch.ones(len(r), dtype=torch.bool))
        reps = eng.sample([ids[i] for i, _, _ in rows], setup, 1.0, args.tokens, args.batch, clock, "generation",
                          S / "_partial_fine_gen.pt")
        v = judge([(T["texts"][i], eng.decode(reps[x][0])) for x, (i, _, _) in enumerate(rows)], args.batch_judge,
                  clock, checkpoint=S / "_partial_fine_judge.pt")
        eng.free()
        return {"sel": sel, "offsets": offs, "rows": rows, "J": [x["J"] for x in v],
                "coherent": [trigram_diversity(r[0]) >= 0.5 for r in reps]}
    Z = phase("fine", S / "fine.pt", redo, clock, _fine)
    rng = np.random.default_rng(args.seed)
    K, N = {}, {}
    for (i, o, _), J, coh in zip(Z["rows"], Z["J"], Z["coherent"]):
        if coh:
            K[(i, o)] = K.get((i, o), 0) + int(J >= 0.5)
            N[(i, o)] = N.get((i, o), 0) + 1
    data = []
    for i in Z["sel"]:
        n = np.array([N.get((i, o), 0) for o in Z["offsets"]], float)
        if (n > 0).all():
            data.append((i, np.array([K.get((i, o), 0) for o in Z["offsets"]], float), n))
    curves = [k / n for _, k, n in data]
    obs = descent(curves)
    isos = [np.clip(isotonic(k / n, n), 0, 1) for _, k, n in data]
    sims = [descent([rng.binomial(n.astype(int), p) / n for (_, _, n), p in zip(data, isos)]) for _ in range(args.n_sim)]
    boot = [descent([curves[j] for j in rng.integers(len(curves), size=len(curves))]) for _ in range(2000)]
    tests = [monotonicity_test(k, n, 500, rng) for _, k, n in data]
    p = np.array([t[0] for t in tests])
    drops = np.array([t[2] for t in tests])
    res = {"n_curves": len(data), "descent_share": obs, "descent_share_ci": [float(np.quantile(boot, .025)),
                                                                           float(np.quantile(boot, .975))],
           "descent_share_noise": float(np.mean(sims)), "descent_share_noise_95": [float(np.quantile(sims, .025)),
                                                                                  float(np.quantile(sims, .975))],
           "p_descent_above_noise": float((1 + np.sum(np.array(sims) >= obs)) / (1 + len(sims))),
           "curves_drop_beyond_noise": float(np.mean(p < 0.05)), "rejections_BH": benjamini_hochberg(p),
           "median_drop": float(np.median(drops)), "drop_q95": float(np.quantile(drops, 0.95))}
    json.dump(res, open(S / "fine.json", "w"), indent=1)
    print("\n" + "=" * 104)
    print(f"(H0) FINE along m: {len(data)} complete curves, offsets {Z['offsets']} around each threshold, "
          f"{args.k} responses per point")
    print(f"  descent share (downward / total variation): {obs * 100:.1f} % [{res['descent_share_ci'][0] * 100:.1f}; "
          f"{res['descent_share_ci'][1] * 100:.1f}], against {res['descent_share_noise'] * 100:.1f} % from noise alone "
          f"(p = {res['p_descent_above_noise']:.3f})")
    print(f"  curves whose largest drop exceeds the noise (p < 0.05): {res['curves_drop_beyond_noise'] * 100:.1f} % "
          f"({res['rejections_BH']} after Benjamini-Hochberg at 10 %)")
    print("=" * 104)
    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axs[0]
    for (i, k, n), pv in zip(data, p):
        rej = pv < 0.05
        ax.plot(Z["offsets"], k / n, color="#D55E00" if rej else "#000000", alpha=0.8 if rej else 0.12, lw=1)
    ax.plot(Z["offsets"], np.mean(curves, 0), color="#CC79A7", lw=2.5, label="mean over prompts")
    ax.axhline(0.5, color="black", ls=":", lw=0.8)
    ax.set(xlabel="offset along m from the threshold of each prompt", ylabel="share of responses judged refusals",
           title=f"{len(data)} prompts, {args.k} responses per point (orange: drop beyond noise)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[1]
    null_drops = []
    for (_, _, n), pp in zip(data, isos):
        for _ in range(20):
            y = rng.binomial(n.astype(int), pp) / n
            null_drops.append(np.max(np.maximum.accumulate(y) - y))
    bins = np.linspace(0, 1, 41)
    ax.hist(drops, bins=bins, density=True, alpha=0.6, color="#000000", label="largest drop of each observed curve")
    ax.hist(null_drops, bins=bins, density=True, histtype="step", lw=2, color="#56B4E9",
            label="largest drop produced by noise alone (simulated increasing curves)")
    ax.set(xlabel="largest drop of a curve, max over t < t' of q(t) - q(t')", ylabel="density",
           title=f"descent share {obs * 100:.1f} % observed, {res['descent_share_noise'] * 100:.1f} % from noise alone")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("Refusal along m around the threshold of each prompt", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(S / "figures" / "h0_fine.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return S / "figures" / "h0_fine.png"


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    for key in cli.model_keys(args):
        S = root / key
        clock = start_run(S, args)
        figs = []
        if args.part in ("local", "both"):
            figs.append(local(key, S, args))
        if args.part in ("fine", "both"):
            figs.append(fine(key, S, args, make_judge(args.judge, dtype_of(args.dtype), args.device), clock,
                             redo_set(args.redo)))
        figures_written(figs)


if __name__ == "__main__":
    main()
