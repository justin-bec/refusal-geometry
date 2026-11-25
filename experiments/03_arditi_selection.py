#!/usr/bin/env python
"""
03_arditi_selection.py -- the direction of Arditi et al., SELECTED by their rule, on each model.
Article: appendix "Protocols" (the direction of Arditi et al.). Prompts of their repository; no attack is generated.

THEIR RULE (Arditi et al., section 2.3 and appendix C).
  candidates   r(l, t) = mu+ - mu- at the input of layer l, at position t of the template (means of
               02_directions.py, or computed here on the same prompts when they are absent, e.g. with
               --directions published), for l < 0.8 L and the P template positions.
  refusal score  log-odds, at the first generated token, of the model's refusal tokens: log p - log(1 - p),
               p = sum of the probabilities of these tokens (configs/models.yaml, refusal_words).
  bypass       mean refusal score over --n_val harmful VALIDATION prompts, the normalized r(l, t) being ablated at
               every layer (block inputs, attention and MLP outputs) and every position;
  induction    mean refusal score over --n_val harmless validation prompts, r(l, t) (not normalized) being ADDED
               at the input of layer l only, at every position;
  KL           mean KL divergence, at the last token, between the intact and the ablated model, on the harmless
               validation prompts.
  selection    among the candidates of induction > 0 and KL < 0.1, the one of lowest bypass score.
  check        the published choices are printed next to ours (Llama-3: layer 12, position -5; Llama-2: layer 14,
               position -1).
All scores need the prefill only (first generated token): no generation here. The direction is evaluated (ablated
everywhere) in 04_ablate_everywhere.py, with the same prompts and protocol as e1.

RULES: candidates and prompts mixed in the same batches (each row carries its direction and its addition layer),
texts sorted by length; no repeated message; the script keeps its own journal; one phase = one file.

USAGE: python experiments/03_arditi_selection.py [--models llama3,gemma]   (after 02_directions.py)
Writes runs/03_arditi_selection/<model>/: journal.log, 01_scores.pt, arditi.pt, figures/selection.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.data import arditi_sets  # noqa: E402
from refusal.generation import first_real  # noqa: E402
from refusal.hooks import Ablation, Addition  # noqa: E402
from refusal.log import figures_written, phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, get_spec  # noqa: E402
from refusal.plots import mpl  # noqa: E402


def parse(argv=None):
    ap = cli.parser("Selection rule of Arditi et al. on each model.")
    ap.add_argument("--n_val", type=int, default=32, help="validation prompts of each kind (their value)")
    ap.add_argument("--layer_share", type=float, default=0.8, help="candidates l < share * L (their value)")
    ap.add_argument("--kl_max", type=float, default=0.1, help="their value")
    ap.add_argument("--batch_score", type=int, default=48)
    ap.add_argument("--n_means", type=int, default=128, help="prompts of each kind for mu+ and mu- (as 02)")
    return ap.parse_args(argv)


def refusal_ids(tok, words):
    out = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t:
            out.append(t[0])
    return sorted(set(out))


def run_model(key, args, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    sets = arditi_sets(args.seed, n_val=args.n_val, n_means=args.n_means)
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed)
    f02 = cli.RUNS / "02_directions" / key / "directions.pt"
    if args.directions == "runs" and f02.exists():
        D = cli.directions(key)
        diff = (D["mu_plus"] - D["mu_minus"]).float()                       # (L, P, d), means of 02_directions
    else:                                                                   # same prompts, same code
        MU = phase("means", S / "00_means.pt", redo, clock, lambda: {
            "mu_plus": eng.mean_states(sets["means_harmful"], args.batch_tf),
            "mu_minus": eng.mean_states(sets["means_harmless"], args.batch_tf)})
        diff = (MU["mu_plus"] - MU["mu_minus"]).float()
    L, P, d = eng.L, eng.P, eng.d
    vh = [eng.encode(t) for t in sets["val_harmful"]]
    vb = [eng.encode(t) for t in sets["val_harmless"]]
    abl = eng.add(Ablation(eng.model, eng.template))
    add = eng.add(Addition(eng.model))
    ref = refusal_ids(eng.tok, spec.refusal_words)
    norms = diff.norm(dim=-1)
    cands = [(l, t) for l in range(L) if l < args.layer_share * L for t in range(P)
             if norms[l, t] > 1e-6 * norms.max()]                 # at layer 0 the template states are the same
                                                                  # embeddings for all prompts: no candidate there
    clock(f"{spec.id}: {len(cands)} candidates (layers < {args.layer_share} L, {P} positions); refusal tokens "
          f"{eng.tok.convert_ids_to_tokens(ref)}")

    def score(lp, row=None):
        lr = torch.logsumexp(lp[ref], -1)
        return float(lr - torch.log1p(-lr.exp().clamp(max=1 - 1e-6)))

    def ablate(rows_cand):
        def f(rows, am):
            U = torch.zeros(len(rows), L, d)
            for b, r in enumerate(rows):
                k = rows_cand(r)
                if k is not None:
                    l, t = cands[k]
                    U[b] = diff[l, t] / diff[l, t].norm()
            abl.set(U=U.to(eng.dev), start=first_real(am))
        return f

    def addition(rows_cand):
        def f(rows, am):
            ks = [rows_cand(r) for r in rows]
            add.set(torch.tensor([cands[k][0] for k in ks]), torch.stack([diff[cands[k][0], cands[k][1]] for k in ks]))
        return f

    def _scores():
        nk = len(cands)
        base_b = eng.first_token_logprobs(vb, None, args.batch_score)
        lh = [(i, k) for k in range(nk) for i in range(len(vh))]
        lb = [(i, k) for k in range(nk) for i in range(len(vb))]
        byp = eng.first_token_logprobs([vh[i] for i, _ in lh], ablate(lambda r: lh[r][1]), args.batch_score, score)
        clock("  bypass scores done")
        ind = eng.first_token_logprobs([vb[i] for i, _ in lb], addition(lambda r: lb[r][1]), args.batch_score, score)
        clock("  induction scores done")
        kl = eng.first_token_logprobs([vb[i] for i, _ in lb], ablate(lambda r: lb[r][1]), args.batch_score,
                                      lambda lp, r: float((base_b[lb[r][0]].to(lp.device).exp()
                                                           * (base_b[lb[r][0]].to(lp.device) - lp)).sum()))
        clock("  KL done")
        mean = lambda vals, rows: np.bincount([k for _, k in rows], weights=vals, minlength=nk) / (len(rows) / nk)  # noqa: E731
        b_, i_, k_ = mean(np.array(byp), lh), mean(np.array(ind), lb), mean(np.array(kl), lb)
        ok = (i_ > 0) & (k_ < args.kl_max)
        chosen = int(np.argmin(np.where(ok, b_, np.inf))) if ok.any() else int(np.argmin(b_))
        none_h = float(np.mean(eng.first_token_logprobs(vh, None, args.batch_score, score)))
        none_b = float(np.mean([score(lp) for lp in base_b]))
        return {"candidates": cands, "bypass": b_, "induction": i_, "kl": k_, "admissible": ok, "chosen": chosen,
                "none_admissible": not bool(ok.any()), "bypass_without": none_h, "induction_without": none_b}

    Sc = phase("scores", S / "01_scores.pt", redo, clock, _scores)
    eng.free()
    l_s, t_s = Sc["candidates"][Sc["chosen"]]
    pos = t_s - P
    pub = spec.arditi_published
    k = Sc["chosen"]
    rule = {"layer": int(l_s), "position": int(pos)}
    if spec.arditi_choice:                               # imposed in configs/models.yaml (see the comment there)
        l_f, p_f = spec.arditi_choice
        if (l_f, p_f + P) not in [tuple(c) for c in Sc["candidates"]]:
            raise SystemExit(f"arditi_choice {spec.arditi_choice} is not a candidate of the rule")
        clock(f"  their rule picks layer {l_s}, position {pos} (bypass {Sc['bypass'][k]:.2f}); "
              f"configs/models.yaml imposes layer {l_f}, position {p_f}")
        k = [tuple(c) for c in Sc["candidates"]].index((l_f, p_f + P))
        l_s, t_s = Sc["candidates"][k]
        pos = t_s - P
    clock(f"  chosen: layer {l_s}, position {pos} (bypass {Sc['bypass'][k]:.2f}, induction {Sc['induction'][k]:.2f}, "
          f"KL {Sc['kl'][k]:.3f}; without intervention {Sc['bypass_without']:.2f} / {Sc['induction_without']:.2f}); "
          f"{int(Sc['admissible'].sum())} admissible candidates"
          + (f"; published: layer {pub[0]}, position {pub[1]} -> {'SAME' if [l_s, pos] == list(pub) else 'DIFFERENT'}"
             if pub else ""))
    if Sc["none_admissible"]:
        clock("  WARNING: no admissible candidate; the lowest bypass score was taken without the filters")
    r = diff[l_s, t_s] / diff[l_s, t_s].norm()
    torch.save({"layer": int(l_s), "position": int(pos), "direction": r, "published": pub}, S / "arditi.pt")
    json.dump({"layer": int(l_s), "position": int(pos), "published": pub, "admissible": int(Sc["admissible"].sum()),
               "candidates": len(Sc["candidates"]), "rule_choice": rule,
               "imposed": bool(spec.arditi_choice)}, open(S / "summary.json", "w"), indent=1)

    plt = mpl()
    nl = max(l for l, _ in Sc["candidates"]) + 1
    Gr = np.full((P, nl), np.nan)
    A = np.zeros((P, nl), bool)
    for j, (l, t) in enumerate(Sc["candidates"]):
        Gr[t, l], A[t, l] = Sc["bypass"][j], Sc["admissible"][j]
    fig, ax = plt.subplots(figsize=(min(18, 4 + 0.35 * nl), 3.8))
    im = ax.imshow(Gr, aspect="auto", cmap="viridis_r", origin="lower")
    xs, ys = np.where(~A.T & ~np.isnan(Gr.T))
    ax.scatter(xs, ys, marker="x", color="white", s=18, label="excluded by the induction or KL filter")
    ax.scatter([l_s], [t_s], marker="*", color="red", s=220, label="direction used")
    if spec.arditi_choice:
        ax.scatter([rule["layer"]], [rule["position"] + P], marker="o", facecolor="none", edgecolor="red", s=160,
                   label="picked by their rule (not used, see configs/models.yaml)")
    ax.set_yticks(range(P))
    ax.set_yticklabels([str(t - P) for t in range(P)])
    ax.set(xlabel="layer", ylabel="position in the template",
           title=f"{spec.id}: bypass score of the candidates of Arditi et al. (lower = better)")
    fig.colorbar(im, ax=ax, label="refusal score under ablation")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(S / "figures" / "selection.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    figures_written([S / "figures" / "selection.png"])


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    for key in cli.model_keys(args):
        run_model(key, args, root)


if __name__ == "__main__":
    main()
