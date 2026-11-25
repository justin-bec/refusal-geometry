#!/usr/bin/env python
"""
08_slopes.py -- the normal to the border at ANCHORS (published prompts taken at their threshold point), estimated
twice independently; the slope of the threshold follows from it.
Article: "A wide region: (H2)" (and the data of "An axis: (H1)").
Llama-3-8B-Instruct by default. Published prompts of 07_thresholds.py; nothing is generated or optimized.

OBJECTS (one prompt = one fixed context).
  coordinates  x = y + u m, y orthogonal to m; the region of prompt n is {u >= u*_n(y)}.
  anchor       a published prompt with a measured (not censored) threshold, taken at its threshold point
               y_n + u*_n m (its state, its context): --per_group anchors per group, drawn at random.
  normal       g = gradient of the judged refusal q with respect to the template state at the anchor, by REINFORCE,
               estimated TWICE on independent sets of --k_grad responses (g^1, g^2).
  slope        w_n = grad u*_n = -(g - <g, m> m) / <g, m> (implicit function theorem); w^1, w^2 from g^1, g^2.

RULES: all responses of all anchors in the same batches, texts sorted by length; the state is injected by a hook at
prefill (and on the template positions when prompt and response are read in one block); no repeated message; the
script keeps its own journal; one phase = one file, the same command resumes.
DURATION (defaults, ~600 anchors x 2 x 40 responses): ~48 000 responses, generation and judgment ~12 h,
backpropagation a few more hours.

USAGE: python experiments/08_slopes.py   (after 07_thresholds.py)
Writes runs/08_slopes/<model>/: journal.log, anchors.json, normals_a/b/c phase files, anchors.pt.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.hooks import GradCapture, Inject  # noqa: E402
from refusal.judge import make_judge  # noqa: E402
from refusal.log import redo_set, start_run  # noqa: E402
from refusal.models import Engine, dtype_of, get_spec  # noqa: E402
from refusal.reinforce import normals  # noqa: E402


def parse(argv=None):
    ap = cli.parser("Normals and slopes of the threshold at anchors.", models=("llama3",))
    ap.add_argument("--per_group", type=int, default=8, help="anchors per group")
    ap.add_argument("--k_grad", type=int, default=40, help="responses per estimate (two estimates)")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--batch_grad", type=int, default=4)
    ap.add_argument("--max_prompt_tokens", type=int, default=1500)
    return ap.parse_args(argv)


def run_model(key, args, judge, root):
    S = root / key
    clock = start_run(S, args)
    redo = redo_set(args.redo)
    spec = get_spec(key)
    T = cli.load("07_thresholds", key, "thresholds.pt")
    m, X, u, ustar = T["m"].float(), T["X"].float(), T["u"], T["ustar"]
    by = {}
    for i in np.where(np.isfinite(ustar))[0]:
        by.setdefault(T["groups"][i], []).append(int(i))
    r = random.Random(args.seed)
    anchors = []
    for g in sorted(by):
        v = by[g][:]
        r.shuffle(v)
        anchors += v[:args.per_group]
    anchors = sorted(anchors)
    json.dump({"anchors": anchors, "groups": [T["groups"][i] for i in anchors]}, open(S / "anchors.json", "w"))
    clock(f"{len(anchors)} anchors in {len(by)} groups ({args.per_group} per group at most)")
    eng = Engine(spec, args.dtype, args.device, args.mem_frac, args.seed, max_prompt_tokens=args.max_prompt_tokens)
    P, c = eng.P, spec.layer
    inj = eng.add(Inject(eng.model, c, eng.template))
    cap = eng.add(GradCapture(eng.model, c, eng.template))
    states = torch.stack([(X[i] + float(ustar[i] - u[i]) * m).reshape(P, -1) for i in anchors])   # y + u* m

    def setup(items, am):
        inj.set(states[items], torch.ones(len(items), dtype=torch.bool))
    N = normals(eng, judge, [eng.encode(T["texts"][i]) for i in anchors], [T["texts"][i] for i in anchors], setup,
                cap, args.k_grad, 2, S, "normals", clock, redo, args.tokens, args.batch, args.batch_grad,
                args.batch_judge)
    eng.free()
    G = N["G"].reshape(2, len(anchors), -1)
    torch.save({"anchors": anchors, "G1": G[0], "G2": G[1], "q": N["q"], "m": m, "states": states.reshape(len(anchors), -1)},
               S / "anchors.pt")
    clock(f"written to {S / 'anchors.pt'} (analyses: 09_h0.py, 12_width.py)")


def main(argv=None):
    args = parse(argv)
    root = cli.out_dir(args, __file__)
    judge = None
    for key in cli.model_keys(args):
        judge = judge or make_judge(args.judge, dtype_of(args.dtype), args.device)
        run_model(key, args, judge, root)


if __name__ == "__main__":
    main()
