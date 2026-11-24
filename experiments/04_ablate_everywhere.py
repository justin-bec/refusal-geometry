#!/usr/bin/env python
"""
04_ablate_everywhere.py -- removing e1 everywhere, as Arditi et al. remove their direction, and at which vertex.
Article: "Removing e1 everywhere, as Arditi et al." Prompts of their repository; no attack is generated.

ABLATION EVERYWHERE: at every layer (block input, attention and MLP outputs), every prompt position and every
generated token. Two vertices:
  vertex 0       h <- h - <u, h> u (Arditi et al.);
  vertex mu-     the affine ablation of Marshall et al.: at the input of layer l, the coordinate along u is brought
                 back to gamma_l(u), its mean over the positions of harmless prompts (first token excepted),
                 h <- h - (<u, h> - gamma_l(u)) u; on the attention and MLP outputs, vertex 0.
CONDITIONS (directions of 02_directions.py and 03_arditi_selection.py), on --n_test harmful TEST prompts never seen:
  no intervention; Arditi et al. everywhere; e1 (vertex 0) everywhere; e1 everywhere except the first token of the
  prompt; e1^{mu-} everywhere at vertex mu-; e1 (vertex 0) everywhere at vertex mu- (inconsistent vertices: the
  vertex of M must be that of the ablation); a random direction everywhere.
MEASURES (refusal.evaluate): harmful responses (judged harmful AND coherent), refusals, incoherent responses,
per-token KL on harmless test prompts; 95 % intervals by bootstrap over prompts.

RULES: all conditions and prompts mixed in the same batches, texts sorted by length; the ablations act on the
generated tokens too (protocol of Arditi et al., intended); no repeated message; the script keeps its own journal;
one phase = one file.

USAGE: python experiments/04_ablate_everywhere.py [--models ...]   (after 02 and 03)
Writes runs/04_ablate_everywhere/<model>/: journal.log, offsets.pt, ablations.pt, ablations_responses.jsonl,
summary.json, figures/ablate_everywhere.png.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.data import arditi_sets  # noqa: E402
from refusal.evaluate import ablations, layer_direction, print_table  # noqa: E402
from refusal.generation import first_real, pad_left, sorted_batches  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import figures_written, phase, redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.plots import ablation_figure  # noqa: E402


def parse(argv=None):
    ap = cli.parser("e1 and the direction of Arditi et al. removed everywhere; vertex 0 and vertex mu-.")
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--n_kl", type=int, default=64)
    ap.add_argument("--samples", type=int, default=2, help="responses per prompt and condition")
    ap.add_argument("--tokens", type=int, default=64)
    return ap.parse_args(argv)


@torch.no_grad()
def harmless_offsets(eng, prompts, dirs, batch):
    """gamma_l(u) for each direction u: mean of <u, h_l> over the real positions (first token excepted) of the
    harmless prompts, h_l = input of block l. Returns (n_dirs, L)."""
    U = torch.stack([u / u.norm() for u in dirs]).float().to(eng.dev)
    tot, n = torch.zeros(len(dirs), eng.L, device=eng.dev), 0
    for rows in sorted_batches(len(prompts), lambda i: len(prompts[i]), batch):
        ids, am, pos = pad_left([prompts[i] for i in rows], eng.pad_id)
        hs = eng.model(input_ids=ids.to(eng.dev), attention_mask=am.to(eng.dev), position_ids=pos.to(eng.dev),
                       output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        m = am.to(eng.dev).bool().clone()
        m[torch.arange(len(rows), device=eng.dev), first_real(am).to(eng.dev)] = False
        for l in range(eng.L):
            tot[:, l] += (torch.einsum("btd,vd->vbt", hs[l].float(), U) * m[None]).sum((1, 2))
        n += int(m.sum())
    return (tot / n).cpu()


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    D, A = cli.directions(key, args.directions), cli.arditi(key)
    sets = arditi_sets(args.seed, n_test=args.n_test, n_kl=args.n_kl)
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed)
    L, d = eng.L, eng.d
    e1, e1_mu = D["e1"].float(), D["e1_mu"].float()
    rand = torch.randn(d, generator=torch.Generator().manual_seed(args.seed + 1))
    off = phase("offsets", S / "offsets.pt", redo, clock, lambda: {"gamma": harmless_offsets(
        eng, [eng.encode(t) for t in sets["means_harmless"]], [e1, e1_mu], args.batch_tf)})["gamma"]
    every = range(L)
    C = [{"name": "none", "label": "no intervention", "family": "no intervention"},
         {"name": "arditi", "label": f"Arditi et al. (layer {A['layer']}, position {A['position']}), everywhere",
          "family": "Arditi et al.", "U": layer_direction(A["direction"], every, L)},
         {"name": "e1", "label": "e1 (vertex 0), everywhere", "family": "e1, vertex 0",
          "U": layer_direction(e1, every, L)},
         {"name": "e1_skip_first", "label": "e1 (vertex 0), everywhere except the first token",
          "family": "e1, vertex 0", "U": layer_direction(e1, every, L), "skip_first": True},
         {"name": "e1mu_affine", "label": "e1^{mu-}, everywhere, at vertex mu-", "family": "e1, vertex mu-",
          "U": layer_direction(e1_mu, every, L), "offset": off[1]},
         {"name": "e1_affine", "label": "e1 (vertex 0), everywhere, at vertex mu- (inconsistent vertices)",
          "family": "inconsistent vertices", "U": layer_direction(e1, every, L), "offset": off[0]},
         {"name": "random", "label": "random direction, everywhere", "family": "random direction",
          "U": layer_direction(rand, every, L)}]
    clock(f"{spec.id}: {len(C)} conditions on {len(sets['test_harmful'])} harmful test prompts; offsets gamma_l(e1) "
          f"at layers 0 / {L // 2} / {L - 1}: {float(off[0, 0]):+.2f} {float(off[0, L // 2]):+.2f} "
          f"{float(off[0, -1]):+.2f}")
    T = ablations(eng, judge, C, sets["test_harmful"], sets["test_harmless"], S, clock, redo, args.samples,
                  args.tokens, args.batch, args.batch_tf, args.batch_judge)
    eng.free()
    print_table(T, f"{spec.id}: ablations everywhere (all layers, all positions, generated tokens)")
    json.dump(T, open(S / "summary.json", "w"), indent=1, ensure_ascii=False)
    f = ablation_figure(S / "figures" / "ablate_everywhere.png", T,
                        f"{spec.id}: directions removed everywhere (all layers, all positions, generated tokens)")
    figures_written([f])


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    judge = None
    for key in cli.model_keys(args):
        judge = judge or make_judge(args.judge, dtype_of(args.dtype), args.device)
        run_model(key, args, judge, root)


if __name__ == "__main__":
    main()
