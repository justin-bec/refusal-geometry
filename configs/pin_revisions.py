#!/usr/bin/env python
"""
pin_revisions.py -- writes in configs/models.yaml the Hugging Face commit of the weights of each model (and of the
judge) found in the LOCAL cache, i.e. the version that was actually used. A direction only makes sense for given
weights: anyone reloading e1 or m must load the same revision.

USAGE (from the repository root, on the machine where the models were downloaded):
    python configs/pin_revisions.py            (prints what it finds and writes it)
    python configs/pin_revisions.py --dry_run  (prints only)
If a model has several versions in the cache, the one the local `main` ref points to is taken.
"""
import argparse
import re
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent / "models.yaml"


def cached_revisions():
    from huggingface_hub import scan_cache_dir
    out = {}
    for repo in scan_cache_dir().repos:
        if repo.repo_type != "model":
            continue
        revs = sorted(repo.revisions, key=lambda r: ("main" not in r.refs, -r.last_modified))
        out[repo.repo_id] = [(r.commit_hash, sorted(r.refs)) for r in revs]
    return out


def main():
    ap = argparse.ArgumentParser(description="Pin the revisions of the models found in the local cache.")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CONFIG))
    cache = cached_revisions()
    text = CONFIG.read_text().splitlines()
    for key, spec in cfg.items():
        revs = cache.get(spec["id"])
        if not revs:
            print(f"{key:8s} {spec['id']}: NOT in the local cache, left as is")
            continue
        sha, refs = revs[0]
        more = f" ({len(revs)} versions in the cache, kept the one of refs {refs or 'none'})" if len(revs) > 1 else ""
        print(f"{key:8s} {spec['id']}: {sha}{more}")
        start = text.index(f"{key}:")
        for j in range(start + 1, len(text)):
            if not text[j].startswith("  "):
                break
            if re.match(r"\s+revision:", text[j]):
                text[j] = f"  revision: {sha}"
                break
    if not a.dry_run:
        CONFIG.write_text("\n".join(text) + "\n")
        print(f"written to {CONFIG}")


if __name__ == "__main__":
    main()
