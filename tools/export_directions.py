#!/usr/bin/env python
"""
export_directions.py -- writes the directions computed by experiments/02_directions.py in the published form:
directions/<model>/directions.safetensors and directions/<model>/metadata.json. CPU, seconds.

PUBLISHED, for each model (unit vectors, float32):
  e1          (d,)     first eigenvector of the per-position M at vertex 0: removed from one layer (any position) or
                       from every layer, as the direction of Arditi et al.;
  e1_concat   (P, d)   first eigenvector of the concatenated M at vertex 0, a unit vector of R^{P d} (the P template
                       positions, in order): removed from the template state of the main layer at prefill;
  m           (P, d)   normalized mean of the unit normals to the border, in R^{P d}: only for the models of --with_m
                       (by default Llama-3-8B-Instruct, the only model where m was tested as an axis).
SIGN CONVENTIONS: <e1, mean over the template positions of (mu+ - mu-) at the main layer> >= 0; <e1_concat, m> >= 0;
m points toward refusal (mean of the normals, which point into the refusal region).
safetensors rather than .pt: a .pt is loaded with pickle, which can execute code.

USAGE (from the repository root, after 02_directions.py):
    python tools/export_directions.py [--models llama3,llama2,qwen,gemma] [--with_m llama3]
Then the experiments accept --directions published.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refusal import cli  # noqa: E402
from refusal.models import get_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Export e1, e1^concat and m in the published form.")
    ap.add_argument("--models", default=",".join(cli.ALL_MODELS))
    ap.add_argument("--with_m", default="llama3")
    ap.add_argument("--out", default=str(cli.PUBLISHED))
    a = ap.parse_args()
    from safetensors.torch import save_file
    with_m = {k.strip() for k in a.with_m.split(",") if k.strip()}
    for key in [k.strip() for k in a.models.split(",") if k.strip()]:
        D = cli.directions(key)
        run = cli.RUNS / "02_directions" / key
        args02 = json.load(open(run / "arguments.json"))
        meta02 = json.load(open(run / "meta.json"))
        stats = json.load(open(run / "summary.json"))["stats"]
        spec = get_spec(key)
        P, d = D["P"], D["d"]
        unit = lambda v: (v / v.norm()).float().contiguous()  # noqa: E731
        T = {"e1": unit(D["e1"].reshape(-1)), "e1_concat": unit(D["e1_concat"].reshape(-1)).reshape(P, d)}
        if key in with_m:
            T["m"] = unit(D["m"].reshape(-1)).reshape(P, d)
        out = Path(a.out) / key
        out.mkdir(parents=True, exist_ok=True)
        f = out / "directions.safetensors"
        save_file(T, str(f), metadata={"model": spec.id, "revision": str(meta02.get("revision"))})
        mc = lambda k: stats[k]  # noqa: E731
        meta = {
            "model": spec.id, "revision": meta02.get("revision"), "layer": D["layer"], "P": P, "d": d, "L": D["L"],
            "template_tokens": meta02["template_tokens"], "template_format": meta02["template_format"],
            "attention": meta02["attention"], "vertex": "0",
            "tensors": {k: list(v.shape) for k, v in T.items()},
            "sign_conventions": "<e1, mean over template positions of (mu+ - mu-) at the main layer> >= 0; "
                                "<e1_concat, m> >= 0; m = normalized mean of the unit normals (points toward refusal)",
            "removal": "e1: h <- h - <e1, h> e1 at the input of a layer and on its attention and MLP outputs; "
                       "e1_concat: x <- x - <x, e1_concat> e1_concat on the template state (P positions "
                       "concatenated) at the input of the main layer, prefill only",
            "border_prompts": stats["n_border"], "folds": stats["folds"],
            "normals": {"responses_per_estimate": args02["k_grad"], "estimates": args02["n_est"]},
            "spectrum": {"e1": {k: mc("per_position|vertex 0")[k] for k in ("ratio_1_2", "share_lambda1",
                                                                            "lambda1_over_rotation", "fold_stability")},
                         "e1_concat": {k: mc("concat|vertex 0")[k] for k in ("ratio_1_2", "share_lambda1",
                                                                             "lambda1_over_rotation", "fold_stability")}},
            "cos_e1_concat_m": stats["cos(e1^concat, m)"],
            "code_commit": args02.get("code_commit"), "seed": args02["seed"],
            "exported": f"{datetime.date.today()}",
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        json.dump(meta, open(out / "metadata.json", "w"), indent=1, ensure_ascii=False)
        print(f"{key}: {', '.join(f'{k} {tuple(v.shape)}' for k, v in T.items())} -> {f}")


if __name__ == "__main__":
    main()
