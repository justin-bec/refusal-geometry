#!/usr/bin/env python
"""
06_one_vector.py -- one vector of R^{P d}, removed from (or added to) the template state of ONE layer, at prefill.
Article: "One vector" (Llama-3-8B-Instruct by default, where the template state of layer 14 is a bottleneck).
Prompts of Arditi et al., MMLU, GSM8K, XSTest (published); no attack is generated or optimized.

OBJECTS (02_directions.py): e1^concat, first eigenvector of the concatenated M at vertex 0, and e1^concat_{mu-},
learned at vertex mu-; e1 (per position). Coupled removal of v from the template state x of layer c at prefill:
x <- x - <x, v> v, ONE scalar product summed over the P positions (a 1-dimensional subspace of R^{P d}).
  (a) REMOVE, on --n_test harmful test prompts: no intervention; e1^concat (vertex 0); e1^concat_{mu-} applied at
      vertex 0 (vertex of M different from that of the ablation); e1 removed from the P template positions of layer
      c (P dimensions); the direction of Arditi et al. removed everywhere (all layers, positions, generated tokens),
      and removed from its own layer only, all positions (the intervention of e1 at one layer).
  (b) ADD, on --n_harmless harmless test prompts: x <- x + t v, with t = |mu+_c - mu-_c|, the norm of the difference
      of the mean harmful and harmless template states (the norm of the direction of Arditi et al. in this space;
      no hand-chosen amplitude); share of coherent refusals, against no intervention.
  (c) CAPABILITIES under no intervention, e1 everywhere (all layers, positions, generated tokens) and e1^concat:
      MMLU, GSM8K (accuracy, NLL of the reference solution), XSTest.

RULES: conditions and prompts mixed in the same batches, texts sorted by length; coupled ablation and injection at
prefill only; no repeated message; the script keeps its own journal; one phase = one file.

USAGE: python experiments/06_one_vector.py [--models llama3]   (after 02 and 03)
Writes runs/06_one_vector/<model>/: journal.log, ablations.pt, addition.pt, capabilities.pt, summary.json,
figures/one_vector.png, figures/addition.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.data import arditi_sets, read_split  # noqa: E402
from refusal.evaluate import (ablations, capabilities, coupled_direction, layer_direction, print_capabilities,  # noqa: E402
                              print_table)
from refusal.generation import trigram_diversity  # noqa: E402
from refusal.hooks import Inject  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import figures_written, phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.plots import ablation_figure, mpl  # noqa: E402
from refusal.stats import bootstrap_ci  # noqa: E402


def parse(argv=None):
    ap = cli.parser("One vector of the template state of one layer: remove, add, capabilities.", models=("llama3",))
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_kl", type=int, default=64)
    ap.add_argument("--n_harmless", type=int, default=100)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n_mmlu", type=int, default=500)
    ap.add_argument("--n_gsm8k", type=int, default=200)
    ap.add_argument("--tokens_gsm8k", type=int, default=256)
    ap.add_argument("--batch_gsm", type=int, default=24)
    return ap.parse_args(argv)


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    D, A = cli.directions(key, args.directions), cli.arditi(key)
    sets = arditi_sets(args.seed, n_test=args.n_test, n_kl=args.n_kl)
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed)
    L, P, c = eng.L, eng.P, spec.layer
    v = D["e1_concat"].float().reshape(-1)
    v = v / v.norm()
    C = [{"name": "none", "label": "no intervention", "family": "no intervention"},
         {"name": "e1_concat", "label": f"e1^concat, template, layer {c}, prefill (one vector)",
          "family": "one vector, one layer", "V": coupled_direction(v, c, L, P)},
         {"name": "e1_concat_mu", "label": "e1^concat learned at vertex mu-, applied at vertex 0",
          "family": "inconsistent vertices", "V": coupled_direction(D["e1_concat_mu"], c, L, P)},
         {"name": "e1_template", "label": f"e1, the {P} template positions, layer {c} ({P} dimensions)",
          "family": "e1, one layer", "U": layer_direction(D["e1"], [c], L), "flags": (False, True, False)},
         {"name": "arditi", "label": f"Arditi et al. (layer {A['layer']}, position {A['position']}), everywhere",
          "family": "Arditi et al.", "U": layer_direction(A["direction"], range(L), L)},
         {"name": "arditi_one_layer", "label": f"Arditi et al., layer {A['layer']} only, all positions",
          "family": "Arditi et al., one layer", "U": layer_direction(A["direction"], [A["layer"]], L)}]
    T = ablations(eng, judge, C, sets["test_harmful"], sets["test_harmless"], S, clock, redo, args.samples,
                  args.tokens, args.batch, args.batch_tf, args.batch_judge)
    print_table(T, f"{spec.id}: one vector removed from the template state of layer {c}")

    # ------------------------------------------------------------------ (b) adding e1^concat to harmless prompts
    def _addition():
        r = np.random.default_rng(args.seed + 3)
        pool = [t for t in read_split("harmless_test") if t not in set(sets["test_harmless"])]
        texts = [pool[i] for i in r.permutation(len(pool))[:args.n_harmless]]
        ids = [eng.encode(t) for t in texts]
        X = eng.template_states(ids, [c], args.batch_tf)[c].reshape(len(ids), -1)
        t = float((D["mu_plus"][c].float() - D["mu_minus"][c].float()).reshape(-1).norm())
        added = X + t * v[None]
        inj = eng.add(Inject(eng.model, c, eng.template))
        rows = [(cond, i, s) for cond in (0, 1) for i in range(len(ids)) for s in range(args.samples)]

        def setup(rr, am):
            inj.set(torch.stack([added[rows[x][1]].reshape(P, -1) for x in rr]),
                    torch.tensor([rows[x][0] == 1 for x in rr]))
        reps = eng.sample([ids[i] for _, i, _ in rows], setup, 1.0, args.tokens, args.batch, clock, "addition")
        inj.remove()
        eng.hooks.remove(inj)
        vs = judge([(texts[i], eng.decode(reps[x][0])) for x, (_, i, _) in enumerate(rows)], args.batch_judge, clock)
        ref = [bool(vs[x]["refusal"]) and trigram_diversity(reps[x][0]) >= 0.5 for x in range(len(rows))]
        out = {"amplitude": t}
        for cond, name in ((0, "none"), (1, "added")):
            out[name] = bootstrap_ci([np.mean([ref[x] for x, rw in enumerate(rows) if rw[0] == cond and rw[1] == i])
                                      for i in range(len(ids))])
        return out
    Ad = phase("addition", S / "addition.pt", redo, clock, _addition)
    print(f"\nADDING e1^concat to {args.n_harmless} harmless test prompts (norm {Ad['amplitude']:.2f}, that of the "
          f"difference of the mean harmful and harmless states): coherent refusals "
          f"{Ad['added'][0] * 100:.1f} % [{Ad['added'][1] * 100:.1f}; {Ad['added'][2] * 100:.1f}], against "
          f"{Ad['none'][0] * 100:.1f} % without intervention")

    # ------------------------------------------------------------------ (c) capabilities
    CK = [C[0], {"name": "e1_everywhere", "label": "e1, everywhere", "family": "e1, vertex 0",
                 "U": layer_direction(D["e1"], range(L), L)}, C[1]]
    K = capabilities(eng, judge, CK, S, clock, redo, args.n_mmlu, args.n_gsm8k, args.tokens_gsm8k, args.tokens,
                     args.batch, args.batch_tf, args.batch_gsm, args.batch_judge, args.seed)
    print_capabilities(K, {cd["name"]: cd["label"] for cd in CK})
    eng.free()
    json.dump({"table": T, "addition": Ad, "capabilities": K}, open(S / "summary.json", "w"), indent=1,
              ensure_ascii=False, default=float)
    figs = [ablation_figure(S / "figures" / "one_vector.png", T,
                            f"{spec.id}: one vector removed from the template state of layer {c} (prefill)")]
    plt = mpl()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for x, (name, lab, col) in enumerate((("none", "no intervention", "#999999"),
                                          ("added", "e1^concat added", "#CC79A7"))):
        a = Ad[name]
        ax.bar(x, a[0], 0.6, color=col, yerr=[[a[0] - a[1]], [a[2] - a[0]]], capsize=4, label=lab)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no intervention", "e1^concat added\n(norm of the difference of means)"])
    ax.set(ylim=(0, 1.02), ylabel="coherent refusals (95 % CI)",
           title=f"{spec.id}, layer {c}: {args.n_harmless} harmless test prompts")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(S / "figures" / "addition.png", dpi=130)
    plt.close(fig)
    figs.append(S / "figures" / "addition.png")
    figures_written(figs)


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    judge = None
    for key in cli.model_keys(args):
        judge = judge or make_judge(args.judge, dtype_of(args.dtype), args.device)
        run_model(key, args, judge, root)


if __name__ == "__main__":
    main()
