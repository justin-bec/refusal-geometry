#!/usr/bin/env python
"""
12_width.py -- a measurable criterion of WIDTH of the family of refusal regions, on the slopes of the threshold.
Article: "The width, measured: (H2)". CPU only, no model (reads 08_slopes.py); seconds.

OBJECTS.
  x = y + u m       template state, m unit axis, y orthogonal to m; region of prompt n: {u >= u*_n(y)}.
  w_n = grad u*_n   slope of the threshold at anchor n, from the normal g_n at its threshold point:
                    w_n = -(g_n - <g_n, m> m) / <g_n, m>, from each of the two independent estimates (w^1, w^2).
                    Anchors kept: <g^1, m> > 0 and <g^2, m> > 0 (the slope is defined).
  criterion         1 + <w_n, w_k> > 0 for n != k: the normal to the border of prompt k is, to first order, a direction
                    of addition for prompt n (1 + <w_n, w_k> = <m - w_n, m - w_k> is the first-order margin gain).
  estimates         cross estimate C_nk = (<w^1_n, w^2_k> + <w^2_n, w^1_k>) / 2, unbiased (independent estimates);
                    a pair is a CONFIRMED violation if <w^1_n, w^1_k> < -1 AND <w^2_n, w^2_k> < -1 (two independent
                    estimates agree), the only kind that cannot be put down to noise.
  robustness        the same without the --exclude share of the anchors of largest slope |w| (border almost parallel
                    to m, <grad q, m> small).

RULES: no model is called; the script keeps its own journal; figure paths printed at the end.

USAGE: python experiments/12_width.py
Writes runs/12_width/<model>/: journal.log, results.json, figures/width.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.log import figures_written, start_run  # noqa: E402
from refusal.plots import mpl  # noqa: E402
from refusal.stats import unit  # noqa: E402


def parse(argv=None):
    ap = cli.parser("Width criterion <w_n, w_k> > -1 on the slopes of the threshold.", models=("llama3",))
    ap.add_argument("--exclude", type=float, default=0.1, help="share of largest slopes left out (robustness)")
    return ap.parse_args(argv)


def analyse(name, W1, W2):
    n = len(W1)
    off = ~np.eye(n, dtype=bool)
    Cm = 0.5 * (W1 @ W2.T + W2 @ W1.T)
    C = Cm[off]
    both = ((W1 @ W1.T)[off] < -1) & ((W2 @ W2.T)[off] < -1)
    np.fill_diagonal(Cm, np.inf)
    worst = Cm.min(1)
    r = {"anchors": n, "pairs": int(C.size), "share_criterion_holds": float((C > -1).mean()),
         "median": float(np.median(C)), "quantiles_0.1_1_5_%": [float(np.quantile(C, p)) for p in (.001, .01, .05)],
         "share_confirmed_violations": float(both.mean()), "confirmed_violations": int(both.sum()),
         "share_anchors_with_a_pair_below_-1": float((worst < -1).mean())}
    print(f"  {name}: {n} anchors, {C.size} pairs; criterion 1 + <w_n, w_k> > 0 holds for "
          f"{r['share_criterion_holds'] * 100:.1f} % of the pairs (median <w_n, w_k> {r['median']:+.2f}); confirmed "
          f"violations (both independent estimates below -1) {r['share_confirmed_violations'] * 100:.2f} % "
          f"({r['confirmed_violations']} pairs)")
    return r, C, worst


def run_model(key, args, root):
    S = root / key
    clock = start_run(S, args)
    An = cli.load("08_slopes", key, "anchors.pt")
    G1, G2 = An["G1"].double().numpy(), An["G2"].double().numpy()
    m = unit(An["m"].double().numpy())
    ok = (G1 @ m > 0) & (G2 @ m > 0)
    G1, G2 = G1[ok], G2[ok]
    W1 = -(G1 - np.outer(G1 @ m, m)) / (G1 @ m)[:, None]
    W2 = -(G2 - np.outer(G2 @ m, m)) / (G2 @ m)[:, None]
    clock(f"{int(ok.sum())} of {ok.size} anchors have a defined slope (<g, m> > 0 in both estimates)")
    if ok.sum() < 3:
        clock("fewer than 3 anchors with a defined slope: nothing to analyse")
        return
    nrm = np.linalg.norm(0.5 * (W1 + W2), axis=1)
    keep = nrm <= np.quantile(nrm, 1 - args.exclude)
    print("\nWIDTH CRITERION")
    res, data = {}, {}
    for name, sel in (("all anchors", np.ones(len(W1), bool)),
                      (f"without the {args.exclude * 100:.0f} % largest slopes", keep)):
        res[name], C, worst = analyse(name, W1[sel], W2[sel])
        data[name] = (C, worst)
    json.dump(res, open(S / "results.json", "w"), indent=1)
    plt = mpl()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for (name, (C, worst)), col in zip(data.items(), ("#0072B2", "#D55E00")):
        ax[0].hist(C, bins=200, histtype="step", density=True, color=col, label=name)
        xs = np.sort(C)
        ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, label=name)
        ax[2].hist(worst, bins=60, histtype="step", color=col, label=name)
    for a in ax:
        a.axvline(-1, color="k", ls="--", lw=1)
        a.legend(fontsize=8, title="dashed: the -1 of the criterion", title_fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set(title="(a) Scalar products between the slopes of two anchors",
              xlabel="<w_n, w_k> (cross estimate of the two independent estimates)", ylabel="density")
    ax[1].set(title="(b) Cumulative distribution (log scale: the tail below -1)", xlabel="<w_n, w_k>",
              ylabel="share of the pairs", yscale="log")
    ax[2].set(title="(c) Worst partner of each anchor", xlabel="min over k != n of <w_n, w_k>",
              ylabel="number of anchors")
    fig.suptitle("Width: the normal to the border of one prompt is a direction of addition for another if "
                 "<w_n, w_k> > -1", fontsize=11)
    fig.tight_layout()
    fig.savefig(S / "figures" / "width.png", dpi=140)
    plt.close(fig)
    figures_written([S / "figures" / "width.png"])


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    for key in cli.model_keys(args):
        run_model(key, args, root)


if __name__ == "__main__":
    main()
