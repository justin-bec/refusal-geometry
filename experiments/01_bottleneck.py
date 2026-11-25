#!/usr/bin/env python
"""
01_bottleneck.py -- is the template state of ONE layer a bottleneck of the refusal decision? One-layer
transplantation on pairs of (harmful, harmless) prompts, on each model; it designates the main layer c.
Article: appendix "Protocols" (choice of c for Qwen and Gemma) and "A space that decides alone" (the
bottleneck of Llama-3-8B at layer 14). Prompts of the repository of Arditi et al.; no attack is generated.

OBJECTS.
  template       the P last tokens, identical for all formatted prompts; state = the P positions at the INPUT of
                 layer l (the rest of the computation, context included, is left unchanged).
  layers         --n_layers layers spread over the depth (rounded linspace(1, L - 2, n)).
  conditions     natural: each prompt as it is;
                 H->B, sufficiency (layer l): harmless prompt j, template state replaced at layer l by that of the
                   associated harmful prompt (the refusal obtained measures the SUFFICIENCY of that state);
                 B->H, necessity (layer l): harmful prompt i, template state replaced by that of the associated
                   harmless prompt (the refusal that REMAINS measures its necessity: 0 if necessary);
                 mu- -> H (layer l): harmful prompt, template state replaced by mu-_l, the mean harmless state.
                 Associations by a fixed random permutation (--seed).
  bottleneck     a layer whose template state is both sufficient and necessary.
  measures       judged refusal (WildGuard, continuous J), harmful responses, incoherent responses (trigram diversity
                 < 1/2: the transplantation can break the model instead of removing the refusal); --k responses per
                 row of --tokens tokens.
  main layer     the layer of largest necessity (lowest remaining refusal); compared with configs/models.yaml.

RULES: all conditions (natural, all layers, both directions) mixed in the same batches, texts sorted by length;
injection hooks at prefill only (one per tested layer, each row modified at its layer only); no repeated message;
the script keeps its own journal; one phase = one file; a model that fails is reported and skipped.

USAGE: python experiments/01_bottleneck.py [--models llama3,qwen] [--n_pairs 20 --tokens 128 for the article's
Llama-3 table]
Writes runs/01_bottleneck/<model>/: journal.log, states.pt, responses.pt, results.json, figures/bottleneck.png;
runs/01_bottleneck/figures/synthesis.png.
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.data import arditi_sets  # noqa: E402
from refusal.generation import trigram_diversity  # noqa: E402
from refusal.hooks import Inject  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import figures_written, phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.plots import mpl  # noqa: E402

KINDS = {"suff": "harmful state in a harmless prompt (sufficiency)",
         "nec": "harmless state in a harmful prompt (necessity)",
         "mu": "mean harmless state mu- in a harmful prompt"}


def parse(argv=None):
    ap = cli.parser("One-layer transplantation of the template state: sufficiency and necessity per layer.")
    ap.add_argument("--n_pairs", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--k", type=int, default=2, help="responses per row")
    ap.add_argument("--tokens", type=int, default=64)
    return ap.parse_args(argv)


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    sets = arditi_sets(args.seed, n_means=args.n_pairs)
    H, B = sets["train"][:args.n_pairs], sets["means_harmless"][:args.n_pairs]
    n = min(len(H), len(B))
    H, B = H[:n], B[:n]
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed)
    L, P = eng.L, eng.P
    layers = sorted(set(int(round(x)) for x in np.linspace(1, L - 2, args.n_layers)))
    texts = H + B
    ids = [eng.encode(t) for t in texts]
    clock(f"{spec.id}: {n} pairs, layers {layers}, template of {P} tokens {eng.template_tokens}")
    St = phase("states", S / "states.pt", redo, clock, lambda: eng.template_states(ids, layers, args.batch_tf))
    mu = {l: St[l][n:].mean(0) for l in layers}
    rng = np.random.default_rng(args.seed)
    pi_h, pi_b = rng.permutation(n), rng.permutation(n)
    rows = [(i, None, None, s) for i in range(2 * n) for s in range(args.k)]
    for l in layers:
        rows += [(n + j, l, "suff", s) for j in range(n) for s in range(args.k)]
        rows += [(i, l, "nec", s) for i in range(n) for s in range(args.k)]
        rows += [(i, l, "mu", s) for i in range(n) for s in range(args.k)]

    def donor(r):
        i, l, kind, _ = r
        if kind == "suff":
            return St[l][int(pi_b[i - n])]
        if kind == "nec":
            return St[l][n + int(pi_h[i])]
        return mu[l]

    def _responses():
        inj = {l: eng.add(Inject(eng.model, l, eng.template)) for l in layers}

        def setup(rr, am):
            for l, h in inj.items():
                act = torch.tensor([rows[x][1] == l for x in rr])
                if act.any():
                    h.set(torch.stack([donor(rows[x]) if rows[x][1] == l else torch.zeros(P, eng.d) for x in rr]), act)
        reps = eng.sample([ids[r[0]] for r in rows], setup, 1.0, args.tokens, args.batch, clock, "generation",
                          S / "_partial_gen.pt")
        for h in inj.values():
            h.remove()
            eng.hooks.remove(h)
        v = judge([(texts[r[0]], eng.decode(reps[x][0])) for x, r in enumerate(rows)], args.batch_judge, clock)
        return {"rows": rows, "J": [x["J"] for x in v], "harmful": [x["harmful"] is True for x in v],
                "incoherent": [trigram_diversity(r[0]) < 0.5 for r in reps]}
    R = phase("responses", S / "responses.pt", redo, clock, _responses)
    eng.free()

    def stats(sel):
        return {k: float(np.mean([R[k][x] for x in sel])) for k in ("J", "harmful", "incoherent")}
    nat_h = stats([x for x, r in enumerate(R["rows"]) if r[1] is None and r[0] < n])
    nat_b = stats([x for x, r in enumerate(R["rows"]) if r[1] is None and r[0] >= n])
    per = {l: {k: stats([x for x, r in enumerate(R["rows"]) if r[1] == l and r[2] == k]) for k in KINDS}
           for l in layers}
    main_layer = min(layers, key=lambda l: per[l]["nec"]["J"])
    res = {"model": spec.id, "L": L, "layers": layers, "natural_harmful": nat_h, "natural_harmless": nat_b,
           "per_layer": per, "layer_of_largest_necessity": main_layer, "configured_layer": spec.layer}
    json.dump(res, open(S / "results.json", "w"), indent=1)
    print("\n" + "=" * 104)
    print(f"{spec.id}: judged refusal (%) after a one-layer transplantation of the template state; natural refusal "
          f"{nat_h['J'] * 100:.0f} % (harmful), {nat_b['J'] * 100:.0f} % (harmless)")
    print(f"  {'layer':>5s} {'H->B provoked':>14s} {'B->H remaining':>15s} {'mu- ->H remaining':>18s} {'incoherent':>11s}")
    for l in layers:
        p = per[l]
        print(f"  {l:5d} {p['suff']['J'] * 100:13.0f}% {p['nec']['J'] * 100:14.0f}% {p['mu']['J'] * 100:17.0f}% "
              f"{max(p[k]['incoherent'] for k in KINDS) * 100:10.0f}%")
    print(f"  layer of largest necessity: {main_layer} (configs/models.yaml: {spec.layer})")
    print("=" * 104)
    plt = mpl()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    depth = np.array(layers) / (L - 1)
    for k, col, ls in (("suff", "#D55E00", "-"), ("nec", "#0072B2", "-"), ("mu", "#0072B2", "--")):
        ax.plot(depth, [per[l][k]["J"] for l in layers], ls, marker="o", color=col, label=KINDS[k])
    ax.axhline(nat_h["J"], color="#555555", ls=":", label="natural refusal of the harmful prompts")
    ax.axvline(spec.layer / (L - 1), color="black", lw=0.6, label=f"configured main layer {spec.layer}")
    ax.set(xlabel="relative depth l / (L - 1) of the transplanted layer", ylabel="judged refusal", ylim=(-0.02, 1.02),
           title=f"{spec.id}: template state transplanted at one layer\n(bottleneck: provoked refusal high AND "
                 f"remaining refusal low)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(S / "figures" / "bottleneck.png", dpi=130)
    plt.close(fig)
    figures_written([S / "figures" / "bottleneck.png", synthesis(root)])


def synthesis(root):
    runs = {f.parent.name: json.load(open(f)) for f in sorted(root.glob("*/results.json"))}
    if not runs:
        return None
    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for k, r in runs.items():
        depth = np.array(r["layers"]) / (r["L"] - 1)
        axs[0].plot(depth, [r["per_layer"][str(l) if str(l) in r["per_layer"] else l]["suff"]["J"] for l in r["layers"]],
                    "-o", label=k)
        axs[1].plot(depth, [r["per_layer"][str(l) if str(l) in r["per_layer"] else l]["nec"]["J"] for l in r["layers"]],
                    "-o", label=k)
    axs[0].set(title="Refusal PROVOKED in harmless prompts by the harmful state (sufficiency)")
    axs[1].set(title="Refusal REMAINING in harmful prompts with the harmless state (necessity)")
    for a in axs:
        a.set(xlabel="relative depth l / (L - 1)", ylabel="judged refusal", ylim=(-0.02, 1.02))
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    (root / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "figures" / "synthesis.png", dpi=130)
    plt.close(fig)
    return root / "figures" / "synthesis.png"


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    judge = None
    for key in cli.model_keys(args):
        judge = judge or make_judge(args.judge, dtype_of(args.dtype), args.device)
        try:
            run_model(key, args, judge, root)
        except Exception as e:  # noqa: BLE001
            print(f"{key}: FAILED, skipped ({type(e).__name__}: {str(e)[:300]})")
            traceback.print_exc()


if __name__ == "__main__":
    main()
