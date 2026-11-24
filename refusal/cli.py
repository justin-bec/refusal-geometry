"""Common command-line options, output folders, and reading the outputs of earlier scripts.

Every script writes to its own folder, runs/<script name>/ (one sub-folder per model when it loops over models),
and only READS the outputs of earlier scripts.
"""
import argparse
import os
from pathlib import Path

import torch

from .models import ROOT, load_specs

RUNS = Path(os.environ.get("REFUSAL_RUNS", ROOT / "runs"))
PUBLISHED = Path(os.environ.get("REFUSAL_DIRECTIONS", ROOT / "directions"))
ALL_MODELS = ("llama3", "llama2", "qwen", "gemma")


def parser(description, models=ALL_MODELS):
    from .log import quiet
    quiet()                                    # before any model is loaded (the judge is loaded first)
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--models", default=",".join(models),
                    help=f"model keys of configs/models.yaml, comma separated (known: {', '.join(load_specs())})")
    ap.add_argument("--judge", default="allenai/wildguard")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem_frac", type=float, default=0.9, help="cap on the GPU memory used by the process")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redo", default="", help="phases to recompute, comma separated")
    ap.add_argument("--batch", type=int, default=32, help="generation batch")
    ap.add_argument("--batch_tf", type=int, default=16, help="teacher-forced reading batch")
    ap.add_argument("--batch_judge", type=int, default=24)
    ap.add_argument("--out", default="", help="output folder (default runs/<script name>)")
    ap.add_argument("--directions", default="runs", choices=["runs", "published"],
                    help="runs: e1, e1^concat, m computed by 02_directions.py; published: those of directions/")
    return ap


def model_keys(args):
    specs = load_specs()
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in specs:
            raise SystemExit(f"unknown model key {k}; known: {', '.join(specs)}")
    return keys


def out_dir(args, script):
    return Path(args.out) if args.out else RUNS / Path(script).stem


def load(script, key, name):
    """File ``name`` written by ``script`` for model ``key`` (runs/<script>/<key>/<name>)."""
    f = RUNS / script / key / name
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run experiments/{script}.py first (for {key})")
    return torch.load(f, map_location="cpu", weights_only=False)


def directions(key, source="runs"):
    """Outputs of 02_directions.py, or with source="published" the published directions (directions/<key>/):
    e1, e1_concat and, where published, m, with P, d and the layer. Scripts that need anything else from
    02_directions.py (means, directions at vertex mu-) say so."""
    if source == "runs":
        return load("02_directions", key, "directions.pt")
    return published_directions(key)


def published_directions(key):
    import json
    from safetensors.torch import load_file
    from .models import get_spec
    d = PUBLISHED / key
    if not (d / "directions.safetensors").exists():
        raise FileNotFoundError(f"{d / 'directions.safetensors'} not found")
    meta = json.load(open(d / "metadata.json"))
    spec = get_spec(key)
    if meta["model"] != spec.id or (spec.revision and meta.get("revision") and meta["revision"] != spec.revision):
        print(f"WARNING: the published directions of {key} were computed on {meta['model']} at revision "
              f"{meta.get('revision')}; configs/models.yaml loads {spec.id} at revision {spec.revision}")
    T = load_file(str(d / "directions.safetensors"))
    out = {"model": meta["model"], "layer": meta["layer"], "P": meta["P"], "d": meta["d"], "published": True,
           "e1": T["e1"], "e1_concat": T["e1_concat"].reshape(-1)}
    if "m" in T:
        out["m"] = T["m"].reshape(-1)
    return _Strict(out, key)


class _Strict(dict):
    """A dict that explains which script to run when a key absent from the published directions is asked for."""

    def __init__(self, d, key):
        super().__init__(d)
        self.key = key

    def __missing__(self, k):
        raise KeyError(f"'{k}' is not among the published directions of {self.key}: run "
                       f"experiments/02_directions.py --models {self.key} (and drop --directions published)")


def arditi(key):
    return load("03_arditi_selection", key, "arditi.pt")
