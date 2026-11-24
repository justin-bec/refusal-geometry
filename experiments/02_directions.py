#!/usr/bin/env python
"""
02_directions.py -- boundary points, normals, the matrix M, and the directions e1, e1^concat and m, on each model.
Article: method section (judged refusal and its gradient, boundary points and normals, the axis m, the removal
direction e1) and the spectrum of the one-vector section. Prompts of the repository of Arditi et al. (published);
no attack is generated or optimized.

OBJECTS (main layer c of configs/models.yaml; template = the P last tokens, identical for all prompts).
  state          template state at the INPUT of layer c, the P positions concatenated (in R^{P d}).
  mu+, mu-       mean template states, at the input of every layer, over 128 harmful and 128 harmless TRAIN prompts.
  transplant.    of dose a: at layers 0..c (not above), x <- (1 - a) x + a mu- at the template positions.
  at the border  harmful train prompt refused naturally (q(a=0) >= 1/2) and unblocked at full dose (q(a=1) < 1/2).
  border point   dose a* found by dichotomy (--rounds rounds of --k_border responses), q(a*) ~ 1/2.
  normal g_n     gradient of the judged refusal q with respect to the state at the border point; REINFORCE on
                 --k_grad responses, leave-one-out baseline; estimated TWICE on independent responses (g^1, g^2).
  natural state  X_n, the state computed by the model without intervention.
  M              M = 1/2 sum (a g^T + g a^T), a = X - O, g = (g^1 + g^2) / 2, vertex O = 0 (that of the ablation
                 applied later) or O = mu- (control). PER POSITION: every (prompt, position) pair is a sample, in
                 R^d -> e1 (and e1^{mu-} at vertex mu-). CONCATENATED: every prompt is a sample, in R^{P d} ->
                 e1^concat. Controls: rotation (normals replaced by random vectors of the same norm) and permutation
                 (state-normal pairing shuffled).
  m              normalized mean of the unit normals g_n / |g_n|, in R^{P d}.
  folds          the border prompts are split in two folds, two rephrasings of one request always in the same fold;
                 directions learned on each fold measure their stability.

RULES: large batches, all prompts mixed, texts sorted by length; transplantation hooks act at prefill only (and on
the template positions when a prompt and its response are read in one block); no repeated message; the script
keeps its own journal; a batch that does not fit is cut in two; one phase = one file, the same command resumes
(--redo to force a phase).

USAGE (from the repository root):
    python experiments/02_directions.py                      (the four models)
    python experiments/02_directions.py --models llama3
Writes runs/02_directions/<model>/: journal.log, meta.json, phase files, directions.pt, summary.json,
figures/spectrum_M.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.data import arditi_sets, duplicate_groups, two_folds  # noqa: E402
from refusal.hooks import GradCapture, Transplant  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import figures_written, phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.plots import mpl, spectrum_panel  # noqa: E402
from refusal.reinforce import normals  # noqa: E402
from refusal.spectrum import low_rank_eig, orient, rotation_null, summary  # noqa: E402


def parse(argv=None):
    ap = cli.parser("Boundary points, normals, M, e1, e1^concat and m.")
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_means", type=int, default=128)
    ap.add_argument("--k_border", type=int, default=6, help="responses per point of the dose dichotomy")
    ap.add_argument("--rounds", type=int, default=4, help="rounds of the dose dichotomy")
    ap.add_argument("--k_grad", type=int, default=48, help="responses per estimate of a normal")
    ap.add_argument("--n_est", type=int, default=2, help="independent estimates of each normal")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n_vec", type=int, default=8)
    ap.add_argument("--n_null", type=int, default=20, help="draws of each null control")
    ap.add_argument("--batch_grad", type=int, default=8)
    return ap.parse_args(argv)


def unit(v):
    return v / v.norm().clamp(min=1e-30)


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    sets = arditi_sets(args.seed, n_train=args.n_train, n_means=args.n_means)
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed)
    c, P, L, d = spec.layer, eng.P, eng.L, eng.d
    json.dump(eng.meta(), open(S / "meta.json", "w"), indent=1, ensure_ascii=False)
    clock(f"{spec.id}: {L} layers, d = {d}, template of {P} tokens {eng.template_tokens}, main layer {c}, "
          f"attention {spec.attention}; {len(sets['train'])} harmful train prompts")
    train = sets["train"]
    ids = [eng.encode(t) for t in train]

    # ------------------------------------------------------------------ mu+ and mu-
    def _means():
        return {"mu_plus": eng.mean_states(sets["means_harmful"], args.batch_tf),
                "mu_minus": eng.mean_states(sets["means_harmless"], args.batch_tf)}
    MU = phase("means", S / "01_means.pt", redo, clock, _means)
    mu_p, mu_m = MU["mu_plus"], MU["mu_minus"]

    tr = eng.add(Transplant(eng.model, c, eng.template))
    cap = eng.add(GradCapture(eng.model, c, eng.template))

    def refusal_at(items, doses, k, title):
        """Mean J of k responses per prompt, each prompt at its transplantation dose."""
        rows = [(i, s) for i in range(len(items)) for s in range(k)]

        def setup(r, am):
            tr.set(torch.tensor([float(doses[rows[x][0]]) for x in r]), mu_m)
        reps = eng.sample([ids[items[i]] for i, _ in rows], setup, 1.0, args.tokens, args.batch, clock, title)
        v = judge([(train[items[i]], eng.decode(reps[x][0])) for x, (i, _) in enumerate(rows)], args.batch_judge)
        J = np.array([x["J"] for x in v]).reshape(len(items), k)
        return J.mean(1)

    # ------------------------------------------------------------------ border points
    def _border():
        n = len(ids)
        q0 = refusal_at(list(range(n)), np.zeros(n), args.k_border, "natural refusal")
        q1 = refusal_at(list(range(n)), np.ones(n), args.k_border, "full dose")
        keep = np.where((q0 >= 0.5) & (q1 < 0.5))[0]
        clock(f"  refused naturally {int((q0 >= 0.5).sum())}/{n}; at the border {len(keep)} "
              f"(blocked: {int(((q0 >= 0.5) & (q1 >= 0.5)).sum())})")
        lo, hi = np.zeros(len(keep)), np.ones(len(keep))
        for t in range(args.rounds):
            mid = 0.5 * (lo + hi)
            q = refusal_at(list(keep), mid, args.k_border, f"dose dichotomy {t + 1}/{args.rounds}")
            lo, hi = np.where(q >= 0.5, mid, lo), np.where(q >= 0.5, hi, mid)
            clock(f"  dose dichotomy round {t + 1}/{args.rounds}: mean refusal at the midpoints {q.mean():.3f}")
        return {"q0": q0, "q1": q1, "keep": keep, "dose": 0.5 * (lo + hi)}
    B = phase("border", S / "02_border.pt", redo, clock, _border)
    keep, dose = [int(i) for i in B["keep"]], np.asarray(B["dose"], float)
    if len(keep) < 4:
        raise RuntimeError(f"only {len(keep)} prompts at the border: nothing to estimate")

    # ------------------------------------------------------------------ normals (two independent estimates)
    def setup_normals(items, am):
        tr.set(torch.tensor([float(dose[i]) for i in items]), mu_m)
    N = normals(eng, judge, [ids[i] for i in keep], [train[i] for i in keep], setup_normals, cap, args.k_grad,
                args.n_est, S, "03_normals", clock, redo, args.tokens, args.batch, args.batch_grad, args.batch_judge)
    G = N["G"].float()                                                     # (n_est, N, P, d)

    # ------------------------------------------------------------------ natural states
    X = phase("natural_states", S / "04_natural_states.pt", redo, clock,
              lambda: {"X": eng.template_states([ids[i] for i in keep], [c], args.batch_tf)[c]})["X"]
    eng.free()

    # ------------------------------------------------------------------ spectra and directions
    folds = np.array(two_folds(duplicate_groups([train[i] for i in keep]), args.seed))
    Gm = G.mean(0)
    n = len(keep)

    def spectra(sel, with_null):
        out = {}
        for vname, O in (("0", torch.zeros(P, d)), ("mu-", mu_m[c])):
            A = X[sel] - O[None]
            for space, (Am, Gg) in {"per_position": (A.reshape(-1, d).T, Gm[sel].reshape(-1, d).T),
                                    "concat": (A.reshape(len(sel), -1).T, Gm[sel].reshape(len(sel), -1).T)}.items():
                w, vecs, null_p = low_rank_eig(Am, Gg, args.n_vec, args.n_null if with_null else 0, args.seed)
                null_r = rotation_null(Am, Gg, args.n_vec, args.n_null, args.seed) if with_null else None
                out[(vname, space)] = {"w": w, "vec": vecs[0], "summary": summary(w.numpy(), null_p, null_r,
                                                                                    args.n_vec)}
        return out

    def _spectra():
        allp = spectra(list(range(n)), True)
        per_fold = [spectra(list(np.where(folds == f)[0]), False) for f in (0, 1)]
        ghat = Gm.reshape(n, -1)
        ghat = ghat / ghat.norm(dim=1, keepdim=True).clamp(min=1e-30)
        m = unit(ghat.mean(0))
        dim_dir = (mu_p[c] - mu_m[c]).mean(0)
        e1 = orient(allp[("0", "per_position")]["vec"], dim_dir)
        e1_mu = orient(allp[("mu-", "per_position")]["vec"], dim_dir)
        e1c = orient(allp[("0", "concat")]["vec"], m)
        e1c_mu = orient(allp[("mu-", "concat")]["vec"], m)
        cosabs = lambda a, b: float(abs(a.reshape(-1) @ b.reshape(-1)) / (a.norm() * b.norm()))  # noqa: E731
        stats = {}
        for key_, s in allp.items():
            r = dict(s["summary"])
            w = s["w"].numpy()
            r["cos_phi_population"] = float((w[0] + w[-1]) / (w[0] - w[-1]))
            r["fold_stability"] = cosabs(per_fold[0][key_]["vec"], per_fold[1][key_]["vec"])
            stats[f"{key_[1]}|vertex {key_[0]}"] = r
        stats["cos(e1 vertex 0, e1 vertex mu-)"] = cosabs(e1, e1_mu)
        stats["cos(e1^concat, m)"] = cosabs(e1c, m)
        stats["n_border"] = n
        stats["folds"] = [int((folds == 0).sum()), int((folds == 1).sum())]
        return {"e1": e1, "e1_mu": e1_mu, "e1_concat": e1c, "e1_concat_mu": e1c_mu, "m": m, "stats": stats,
                "spectra": {f"{k[1]}|vertex {k[0]}": v["summary"] for k, v in allp.items()}}
    D = phase("spectra", S / "05_spectra.pt", redo, clock, _spectra)
    st = D["stats"]
    print("\n" + "=" * 104)
    print(f"{spec.id}, layer {c}: spectrum of M ({n} border prompts, folds of {st['folds'][0]} and {st['folds'][1]})")
    print(f"  {'M':28s} {'l1/l2':>7s} {'share l1':>9s} {'l1/rot.':>8s} {'l1/perm.':>9s} {'>rot.':>6s} "
          f"{'fold cos':>9s} {'cos phi':>8s}")
    for k_, r in st.items():
        if isinstance(r, dict):
            print(f"  {k_:28s} {r['ratio_1_2']:7.2f} {r['share_lambda1'] * 100:8.1f}% "
                  f"{r['lambda1_over_rotation']:8.1f} {r['lambda1_over_permutation']:9.2f} {r['above_rotation']:6d} "
                  f"{r['fold_stability']:9.3f} {r['cos_phi_population']:8.3f}")
    print(f"  |cos(e1 vertex 0, e1 vertex mu-)| = {st['cos(e1 vertex 0, e1 vertex mu-)']:.3f} ; "
          f"|cos(e1^concat, m)| = {st['cos(e1^concat, m)']:.3f}")
    print("  share l1: lambda_1 / sum of positive eigenvalues (share of the largest first-order drop of sum q that one")
    print("  direction obtains); l1/rot., l1/perm.: lambda_1 over the mean lambda_1 of each control; >rot.: eigenvalues")
    print("  above the maximum of the rotation control; cos phi: cosine between mean state and mean normal if the")
    print("  covariance part of M is negligible, (l1 + l_min) / (l1 - l_min)")
    print("=" * 104)

    out = {"model": spec.id, "layer": c, "P": P, "d": d, "L": L, "mu_plus": mu_p, "mu_minus": mu_m,
           "e1": D["e1"], "e1_mu": D["e1_mu"], "e1_concat": D["e1_concat"], "e1_concat_mu": D["e1_concat_mu"],
           "m": D["m"], "X": X, "G": G, "q_border": N["q"], "train_index": keep, "folds": folds,
           "border_prompts": [train[i] for i in keep]}
    torch.save(out, S / "directions.pt")
    json.dump({"stats": st, "spectra": D["spectra"]}, open(S / "summary.json", "w"), indent=1)

    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, sp, title in ((axs[0], "per_position|vertex 0", "per position (e1, a direction of R^d), vertex 0"),
                          (axs[1], "concat|vertex 0", "concatenated (e1^concat, a direction of R^{Pd}), vertex 0")):
        r = st[sp]
        spectrum_panel(ax, D["spectra"][sp], f"{title}\nlambda1/lambda2 = {r['ratio_1_2']:.1f}, "
                       f"cos between folds {r['fold_stability']:.3f}", args.n_vec)
    fig.suptitle(f"{spec.id}, layer {c}: spectrum of M = 1/2 sum (a g^T + g a^T), states a and normals g at the "
                 f"border ({n} prompts)\nbands: 5-95 % of the two controls", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(S / "figures" / "spectrum_M.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    figures_written([S / "figures" / "spectrum_M.png"])
    clock(f"{key}: done")


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    judge = None
    for key in cli.model_keys(args):
        if judge is None:
            judge = make_judge(args.judge, dtype_of(args.dtype), args.device)
        run_model(key, args, judge, root)


if __name__ == "__main__":
    main()
