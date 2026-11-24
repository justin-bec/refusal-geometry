#!/usr/bin/env python
"""
05_one_layer.py -- removing e1 at ONE layer: which positions matter, at which layers it works, and the capabilities.
Article: "One layer, all positions". Prompts of Arditi et al., MMLU, GSM8K, XSTest (published); no attack generated.

POSITIONS of a formatted prompt: INSTRUCTION = the real positions before the template (including the begin token
and the user header); TEMPLATE = the P last prompt positions; GENERATED = the response tokens. Padding is never
touched. Ablation at vertex 0, at the input of the layer and on the outputs of its attention and MLP, on the chosen
positions only.
CONDITIONS (main layer c of configs/models.yaml; e1 and e1^concat of 02_directions.py):
  (a) at layer c, e1 removed from: all positions; the prompt (instruction + template); the instruction; the
      template; the generated tokens. e1^concat removed from the template state at prefill ("one vector, one
      layer"). A random direction, all positions (control). The direction of Arditi et al. removed EVERYWHERE (all
      layers), for comparison: it is the stronger intervention.
  (b) the same e1, learned at layer c, removed from all positions of ONE other layer l, for 6 layers spread over
      the depth (rounded linspace(0, L - 1, 8) without its ends).
MEASURES (refusal.evaluate) on --n_test harmful test prompts; KL on --n_kl harmless ones. Capabilities for "no
intervention", "e1, layer c, all positions" and "e1, layer c, prompt only": MMLU (--n_mmlu questions), GSM8K
(--n_gsm8k problems, greedy answers of --tokens_gsm8k tokens; NLL of the reference solution), XSTest.

RULES: all conditions and prompts mixed in the same batches, texts sorted by length; the ablations that touch the
generated tokens act during generation too (intended); no repeated message; the script keeps its own journal; one
phase = one file, the same command resumes; a model that fails is reported and skipped.

USAGE: python experiments/05_one_layer.py [--models ...]   (after 02 and 03)
Writes runs/05_one_layer/<model>/: journal.log, ablations.pt, capabilities.pt, summary.json, figures/one_layer.png;
and runs/05_one_layer/figures/synthesis.png (all the models measured so far).
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
from refusal.evaluate import (ablations, capabilities, coupled_direction, layer_direction, print_capabilities,  # noqa: E402
                              print_table)
from refusal.judge import make_judge  # noqa: E402
from refusal.log import figures_written, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.plots import ablation_figure, mpl  # noqa: E402

ALL, PROMPT, INSTR, TPL, GEN = (1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
MAIN = ("all", "prompt", "instruction", "template", "generated", "e1_concat", "random", "arditi_one_layer", "arditi")


def parse(argv=None):
    ap = cli.parser("e1 at one layer: positions, layers, capabilities.")
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_kl", type=int, default=64)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n_mmlu", type=int, default=200)
    ap.add_argument("--n_gsm8k", type=int, default=100)
    ap.add_argument("--tokens_gsm8k", type=int, default=512)
    ap.add_argument("--batch_gsm", type=int, default=24)
    ap.add_argument("--no_capabilities", action="store_true")
    return ap.parse_args(argv)


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    D, A = cli.directions(key, args.directions), cli.arditi(key)
    sets = arditi_sets(args.seed, n_test=args.n_test, n_kl=args.n_kl)
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed)
    L, P, d, c = eng.L, eng.P, eng.d, spec.layer
    e1 = D["e1"].float()
    rand = torch.randn(d, generator=torch.Generator().manual_seed(args.seed + 1))
    grid = sorted(set(int(x) for x in np.round(np.linspace(0, L - 1, 8))[1:-1]) | {c})
    one = lambda u, layers: layer_direction(u, layers, L)  # noqa: E731
    C = [{"name": "none", "label": "no intervention", "family": "no intervention"},
         {"name": "all", "label": f"e1, layer {c}, all positions", "family": "e1, one layer", "U": one(e1, [c]),
          "flags": ALL},
         {"name": "prompt", "label": f"e1, layer {c}, prompt (instruction + template)", "family": "e1, one layer",
          "U": one(e1, [c]), "flags": PROMPT},
         {"name": "instruction", "label": f"e1, layer {c}, instruction only", "family": "e1, one layer",
          "U": one(e1, [c]), "flags": INSTR},
         {"name": "template", "label": f"e1, layer {c}, template only", "family": "e1, one layer",
          "U": one(e1, [c]), "flags": TPL},
         {"name": "generated", "label": f"e1, layer {c}, generated tokens only", "family": "e1, one layer",
          "U": one(e1, [c]), "flags": GEN},
         {"name": "e1_concat", "label": f"e1^concat, template, layer {c}, prefill (one vector)",
          "family": "one vector, one layer", "V": coupled_direction(D["e1_concat"], c, L, P)},
         {"name": "random", "label": f"random direction, layer {c}, all positions", "family": "random direction",
          "U": one(rand, [c])},
         {"name": "arditi", "label": f"Arditi et al. (layer {A['layer']}, position {A['position']}), everywhere",
          "family": "Arditi et al.", "U": one(A["direction"], range(L))},
         {"name": "arditi_one_layer", "label": f"Arditi et al., layer {A['layer']} only, all positions",
          "family": "Arditi et al., one layer", "U": one(A["direction"], [A["layer"]]), "flags": ALL}]
    for l in grid:
        if l != c:
            C.append({"name": f"layer_{l}", "label": f"e1, layer {l}, all positions", "family": "e1, other layer",
                      "U": one(e1, [l]), "layer": l})
    clock(f"{spec.id}: main layer {c}, {L} layers, sweep {grid}; {len(C)} conditions")
    T = ablations(eng, judge, C, sets["test_harmful"], sets["test_harmless"], S, clock, redo, args.samples,
                  args.tokens, args.batch, args.batch_tf, args.batch_judge)
    print_table(T, f"{spec.id}: e1 removed at ONE layer")
    K = None
    if not args.no_capabilities:
        K = capabilities(eng, judge, [C[0], C[1], C[2]], S, clock, redo, args.n_mmlu, args.n_gsm8k,
                         args.tokens_gsm8k, args.tokens, args.batch, args.batch_tf, args.batch_gsm, args.batch_judge,
                         args.seed)
        print_capabilities(K, {cd["name"]: cd["label"] for cd in C})
    eng.free()
    sweep = {"layers": grid, "L": L, "c": c,
             "harmful": [T["all" if l == c else f"layer_{l}"]["harmful"] for l in grid]}
    json.dump({"table": T, "capabilities": K, "sweep": sweep}, open(S / "summary.json", "w"), indent=1,
              ensure_ascii=False, default=float)
    figs = [ablation_figure(S / "figures" / "one_layer.png", {n: T[n] for n in T if not n.startswith("layer_")},
                            f"{spec.id}: e1 removed at layer {c} only, by type of position", separator_after=None)]
    figs.append(synthesis(root))
    figures_written(figs)


def synthesis(root):
    """(a) the main conditions on every model measured so far; (b) the layer sweep against relative depth."""
    plt = mpl()
    runs = {}
    for f in sorted(root.glob("*/summary.json")):
        runs[f.parent.name] = json.load(open(f))
    if not runs:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    labels = {"all": "e1, all positions", "prompt": "e1, prompt", "instruction": "e1, instruction",
              "template": "e1, template", "generated": "e1, generated tokens", "e1_concat": "e1^concat, template",
              "random": "random direction", "arditi_one_layer": "Arditi et al., its layer only (same intervention as e1)",
              "arditi": "Arditi et al., everywhere (all layers)"}
    cols = ("#009E73", "#117733", "#88CCEE", "#56B4E9", "#0072B2", "#CC79A7", "#555555", "#E69F00", "#D55E00")
    w = 0.8 / len(MAIN)
    xs = np.arange(len(runs))
    for j, (name, col) in enumerate(zip(MAIN, cols)):
        v = np.array([runs[k]["table"][name]["harmful"] for k in runs])
        ax.bar(xs + (j - len(MAIN) / 2 + 0.5) * w, v[:, 0], w, color=col, label=labels[name])
        ax.errorbar(xs + (j - len(MAIN) / 2 + 0.5) * w, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                    fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k} (layer {runs[k]['sweep']['c']})" for k in runs])
    ax.set(ylim=(0, 1.02), ylabel="harmful responses (95 % CI)",
           title="(a) At the main layer c, by the positions e1 is removed from (Arditi et al.: all layers)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    ax = axs[1]
    for k, r in runs.items():
        sw = r["sweep"]
        depth = np.array(sw["layers"]) / (sw["L"] - 1)
        h = np.array(sw["harmful"])
        line = ax.plot(depth, h[:, 0], "-o", label=k)[0]
        ax.fill_between(depth, h[:, 1], h[:, 2], color=line.get_color(), alpha=0.15)
        ax.scatter([sw["c"] / (sw["L"] - 1)], [h[sw["layers"].index(sw["c"]), 0]], marker="*", s=200,
                   color=line.get_color())
    ax.set(xlabel="relative depth l / (L - 1) of the layer e1 is removed from", ylabel="harmful responses",
           ylim=(0, 1.02), title="(b) e1 learned at layer c (star), removed from all positions of one layer l")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("e1 removed at ONE layer (vertex 0): harmful responses to harmful test prompts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    (root / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "figures" / "synthesis.png", dpi=130, bbox_inches="tight")
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
