#!/usr/bin/env python
"""
11_threshold_analysis.py -- is the natural refusal read on the position u, on the threshold u*, or on both? And what
does an attack move? Article: "A threshold per prompt". CPU only, no model (reads 07_thresholds.py); seconds.

OBJECTS (07_thresholds.py).
  u = <X_n, m>      position of the natural state of prompt n along m;
  u*_n              its threshold (dichotomy along m, in its context); CENSORED if stuck at a bound of the grid;
  refused           natural refusal q_n > 1/2.
SCORES of the natural refusal (AUC: probability that a refused prompt drawn at random has a higher score than a
non-refused one; 95 % intervals by bootstrap over prompts):
  u (position only); -u* (threshold only: a low threshold makes refuse); -u* at fixed position (AUC within
  quintiles of u, weighted average); u - u* (margin: partly true by construction, since under (H1) the sign of the
  margin IS the refusal; the informative comparison is between the first two).
  Set (A): censored thresholds excluded; set (B): censored thresholds placed just beyond the bound they touch
  (correct ranks, arbitrary values: only AUCs without strata on u* are meaningful there).
ATTACKS: for a pair (base b, attack a), with u*(s, c) the threshold of the state of s in the context of c:
  D = u*(a, a) - u*(b, b)   the border moves;   C = u*(b, a) - u*(b, b)   part of the CONTEXT of the attack;
  S = u*(a, b) - u*(b, b)   part of the STATE;   I = D - C - S   interaction.
  On one measure, the correlations of D with C and S are biased by the threshold noise that D, C and S share (they all
  contain u*(b, b)). With independent remeasures r != r', mean cov(X_r, Y_r') estimates the true covariance without
  bias, hence the true correlations and the true share of the variance of D carried by I.
  The state moves too: Delta u = u_a - u_b against D, on the pairs whose two own thresholds are measured.

RULES: no model is called; the script keeps its own journal; every result is printed as soon as computed; figures
at the end, with their paths.

USAGE: python experiments/11_threshold_analysis.py
Writes runs/11_threshold_analysis/<model>/: journal.log, results.json, figures/threshold_cloud.png,
figures/correlations_debiased.png, figures/thresholds_attacks.png.
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.log import figures_written, start_run  # noqa: E402
from refusal.plots import mpl  # noqa: E402
from refusal.stats import auc_ci, conditional_auc  # noqa: E402


def parse(argv=None):
    ap = cli.parser("Analysis of the thresholds: AUCs, decomposition of what an attack moves.", models=("llama3",))
    ap.add_argument("--n_boot", type=int, default=2000)
    return ap.parse_args(argv)


def pair_terms(thr, pairs):
    """(D, C, S) arrays over pairs from one measure {(state, ctx): threshold}."""
    out = []
    for p in pairs:
        b, a = p["base"], p["attack"]
        v = [thr.get(k, np.nan) for k in ((a, a), (b, b), (b, a), (a, b))]
        out.append((v[0] - v[1], v[2] - v[1], v[3] - v[1]))
    return np.array(out, float).reshape(-1, 3)


def run_model(key, args, root):
    S = root / key
    clock = start_run(S, args)
    T = cli.load("07_thresholds", key, "thresholds.pt")
    u, us, q = np.asarray(T["u"], float), np.asarray(T["ustar"], float), np.asarray(T["q"], float)
    y = q > 0.5
    lo_b, hi_b = T["u_bounds"]
    M0 = T["measures"][0]
    own = {kk: (lo, hi) for kk, lo, hi in zip(M0["keys"], M0["lo"], M0["hi"])}
    res = {}
    A = np.isfinite(us)
    clock(f"{len(us)} prompts, {int(A.sum())} thresholds measured, {int((~A).sum())} censored; "
          f"{int(y.sum())} refused naturally")
    print("\nAUC OF THE NATURAL REFUSAL (95 % CI, bootstrap over prompts)")
    res["A"] = {"u": auc_ci(u[A], y[A], args.n_boot), "-u*": auc_ci(-us[A], y[A], args.n_boot),
                "margin u-u*": auc_ci(u[A] - us[A], y[A], args.n_boot),
                "-u* at fixed u (quintiles of u)": conditional_auc(-us[A], y[A], u[A]),
                "u at fixed u* (quintiles of u*)": conditional_auc(u[A], y[A], us[A])}
    for k, v in res["A"].items():
        print(f"  (A) measured thresholds  {k:34s} " + (f"{v[0]:.3f} [{v[1]:.3f}; {v[2]:.3f}]" if isinstance(v, tuple)
                                                         else f"{v:.3f}"))
    usB = us.copy()
    for i in np.where(~A)[0]:
        lo, hi = own[(i, i)]
        usB[i] = lo_b - 1e-3 if lo <= lo_b + 1e-9 else hi_b + 1e-3
    res["B"] = {"u": auc_ci(u, y, args.n_boot), "-u*": auc_ci(-usB, y, args.n_boot)}
    for k, v in res["B"].items():
        print(f"  (B) censored placed beyond their bound  {k:19s} {v[0]:.3f} [{v[1]:.3f}; {v[2]:.3f}]")
    gm = {}
    for g in set(T["groups"]):
        sel = A & (np.array(T["groups"]) == g)
        if sel.sum() >= 3:
            gm[g] = (float(np.median(us[sel])), float(np.median(u[sel])))
    res["spread"] = {"sd_u": float(u[A].std()), "sd_u*": float(us[A].std()),
                     "corr_u_u*": float(np.corrcoef(u[A], us[A])[0, 1]),
                     "group_median_u*_range": [min(v[0] for v in gm.values()), max(v[0] for v in gm.values())],
                     "group_median_u_range": [min(v[1] for v in gm.values()), max(v[1] for v in gm.values())]}
    r = res["spread"]
    print(f"\nSPREAD: sd of u* {r['sd_u*']:.2f} against {r['sd_u']:.2f} for u; corr(u, u*) {r['corr_u_u*']:+.2f}; "
          f"group medians of u* from {r['group_median_u*_range'][0]:+.1f} to {r['group_median_u*_range'][1]:+.1f}, "
          f"of u from {r['group_median_u_range'][0]:+.1f} to {r['group_median_u_range'][1]:+.1f}")

    # ------------------------------------------------------------------ attacks
    pairs = T["pairs"]
    meas = [{kk: t for kk, t in zip(M["keys"], M["threshold"])} for M in T["measures"]]
    one = pair_terms(meas[0], pairs)
    ok1 = np.isfinite(one).all(1)
    Dd, Cc, Ss = one[ok1].T
    Ii = Dd - Cc - Ss
    res["pairs_one_measure"] = {"n": int(ok1.sum()), "corr_D_C": float(np.corrcoef(Dd, Cc)[0, 1]),
                                "corr_D_S": float(np.corrcoef(Dd, Ss)[0, 1]),
                                "share_var_I": float(Ii.var() / Dd.var())}
    r = res["pairs_one_measure"]
    print(f"\nPAIRS, one measure ({r['n']} pairs): corr(D, C) {r['corr_D_C']:.2f}, corr(D, S) {r['corr_D_S']:.2f}, "
          f"var(I) / var(D) {r['share_var_I']:.2f} (biased by the shared noise)")
    if len(meas) >= 2:
        terms = [pair_terms(mz, pairs) for mz in meas]
        ok = np.all([np.isfinite(t).all(1) for t in terms], 0)
        Z = [np.column_stack([t[ok], t[ok][:, 0] - t[ok][:, 1] - t[ok][:, 2]]) for t in terms]   # D, C, S, I

        def cov(i, j):
            return float(np.mean([np.cov(Z[a][:, i], Z[b][:, j])[0, 1] for a, b in permutations(range(len(Z)), 2)]))
        var = {k: cov(k, k) for k in range(4)}
        res["pairs_debiased"] = {"n": int(ok.sum()), "measures": len(Z),
                                 "corr_D_C": cov(0, 1) / np.sqrt(var[0] * var[1]),
                                 "corr_D_S": cov(0, 2) / np.sqrt(var[0] * var[2]),
                                 "share_var_I": var[3] / var[0]}
        r = res["pairs_debiased"]
        print(f"PAIRS, debiased ({r['n']} pairs measured {r['measures']} times): corr(D, C) {r['corr_D_C']:.2f}, "
              f"corr(D, S) {r['corr_D_S']:.2f}, var(I) / var(D) {r['share_var_I']:.2f}")
    else:
        print("PAIRS, debiased: no remeasure yet (07_thresholds.py --remeasures 2)")
    du, dD = [], []
    for p in pairs:
        b, a = p["base"], p["attack"]
        if np.isfinite(us[a]) and np.isfinite(us[b]):
            du.append(u[a] - u[b])
            dD.append(us[a] - us[b])
    du, dD = np.array(du), np.array(dD)
    res["state_vs_border"] = {"n": int(len(du)), "mean_delta_u": float(du.mean()), "mean_D": float(dD.mean()),
                              "sd_delta_u": float(du.std()), "sd_D": float(dD.std()),
                              "corr": float(np.corrcoef(du, dD)[0, 1])}
    r = res["state_vs_border"]
    print(f"STATE AND BORDER ({r['n']} pairs, one measure): the attack moves the state by {r['mean_delta_u']:+.2f} on "
          f"average along m and the threshold by {r['mean_D']:+.2f}; standard deviations {r['sd_delta_u']:.2f} and "
          f"{r['sd_D']:.2f}; correlation {r['corr']:+.2f}")
    json.dump(res, open(S / "results.json", "w"), indent=1)

    # ------------------------------------------------------------------ figures
    plt = mpl()
    figs = []
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(u[A & ~y], us[A & ~y], s=8, color="#0072B2", alpha=0.5, label="not refused")
    ax.scatter(u[A & y], us[A & y], s=8, color="#D55E00", alpha=0.5, label="refused")
    lim = [min(u[A].min(), us[A].min()), max(u[A].max(), us[A].max())]
    ax.plot(lim, lim, "k--", lw=1, label="u = u* (refused below this line)")
    ax.set(xlabel="position u of the natural state along m", ylabel="threshold u* of the prompt",
           title=f"{int(A.sum())} published prompts with a measured threshold\nAUC: u {res['A']['u'][0]:.3f}, "
                 f"-u* {res['A']['-u*'][0]:.3f}, margin {res['A']['margin u-u*'][0]:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(S / "figures" / "threshold_cloud.png", dpi=140)
    plt.close(fig)
    figs.append(S / "figures" / "threshold_cloud.png")
    if "pairs_debiased" in res:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        lab = ["corr(D, context C)", "corr(D, state S)", "share of var(D) carried by I"]
        k_ = ["corr_D_C", "corr_D_S", "share_var_I"]
        xs = np.arange(3)
        ax.bar(xs - 0.2, [res["pairs_one_measure"][k] for k in k_], 0.4, color="#999999", label="one measure")
        ax.bar(xs + 0.2, [res["pairs_debiased"][k] for k in k_], 0.4, color="#0072B2",
               label=f"debiased ({res['pairs_debiased']['measures']} independent measures)")
        ax.set_xticks(xs)
        ax.set_xticklabels(lab)
        ax.set(ylim=(0, 1.02), title="What an attack moves: decomposition of the shift D of the threshold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(S / "figures" / "correlations_debiased.png", dpi=140)
        plt.close(fig)
        figs.append(S / "figures" / "correlations_debiased.png")
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].scatter(u, q, s=6, alpha=0.4, color="#000000")
    axs[0].set(xlabel="position u along m", ylabel="natural refusal q", title="(a) Refusal against u: no common threshold")
    axs[1].scatter(u[A] - us[A], q[A], s=6, alpha=0.4, color="#000000")
    axs[1].axvline(0, color="#D55E00", ls="--")
    axs[1].set(xlabel="margin u - u*", ylabel="natural refusal q", title="(b) Refusal against the margin: a step at 0")
    sc = axs[2].scatter(Cc, Ss, c=Dd, cmap="coolwarm", s=10)
    fig.colorbar(sc, ax=axs[2], label="shift D of the threshold")
    axs[2].axhline(0, color="gray", lw=0.6)
    axs[2].axvline(0, color="gray", lw=0.6)
    axs[2].set(xlabel="part of the context C", ylabel="part of the state S",
               title=f"(c) {len(Dd)} (base, attack) pairs, one measure")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(S / "figures" / "thresholds_attacks.png", dpi=130)
    plt.close(fig)
    figs.append(S / "figures" / "thresholds_attacks.png")
    figures_written(figs)


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    for key in cli.model_keys(args):
        run_model(key, args, root)


if __name__ == "__main__":
    main()
