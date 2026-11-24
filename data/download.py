#!/usr/bin/env python
"""
download.py -- fetches every PUBLISHED dataset the experiments read, at pinned revisions, and records what was
fetched (source, revision, file hashes) in data/MANIFEST.json. Nothing is generated or optimized here, and nothing
from these datasets is redistributed in this repository.

  arditi        the train / val / test splits of Arditi et al. (github.com/andyrdt/refusal_direction), the prompts
                of the four-model experiments                                   -> data/arditi/splits/
  capabilities  MMLU (cais/mmlu, test), GSM8K (openai/gsm8k, main, test), XSTest -> data/capabilities/
  published     jailbreak artifacts and benchmarks of the threshold experiments  -> data/published/<dataset>/
                  jailbreakbench   JBB-Behaviors (Hugging Face) and the published attack artifacts (GitHub)
                  wildjailbreak    allenai/wildjailbreak     (GATED: request access on its Hugging Face page)
                  sorry_bench      sorry-bench/sorry-bench-202406 (GATED: request access on its page)
                  or_bench         bench-llm/or-bench
                  xstest           paul-rottger/exaggerated-safety (GitHub)
                  harmbench        centerforaisafety/HarmBench behaviors (GitHub)

GitHub sources are pinned to a commit below. Hugging Face datasets are pinned by HF_REVISIONS; a None revision means
"the current one", and the commit actually downloaded is written to MANIFEST.json so that it can be pinned.
Each dataset is attempted independently: a failure (access not granted, renamed file) is reported and does not stop
the others.

PREREQUISITES: pip install -r requirements.txt; `hf auth login` once (token from huggingface.co/settings/tokens),
and accept the access conditions of the two gated datasets on their pages.

USAGE (from the repository root):
    python data/download.py                      (everything)
    python data/download.py --only arditi,capabilities
"""
import argparse
import hashlib
import json
import sys
import traceback
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent
ARDITI_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
XSTEST_COMMIT = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
HARMBENCH_COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
JBB_ARTIFACTS_COMMIT = "909e68c01d94222b8ad2e397a017e2e12e2adb73"
HF_REVISIONS = {"cais/mmlu": None, "openai/gsm8k": None, "JailbreakBench/JBB-Behaviors": None,
                "allenai/wildjailbreak": None, "sorry-bench/sorry-bench-202406": None, "bench-llm/or-bench": None}
ARDITI_SPLITS = ("harmful_train", "harmful_val", "harmful_test", "harmless_train", "harmless_val", "harmless_test")
MANIFEST = {}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def record(name, source, revision, files):
    MANIFEST[name] = {"source": source, "revision": revision,
                      "files": {str(Path(f).relative_to(DATA)): sha256(f) for f in files}}


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def hf_revision(repo, revision):
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo, revision=revision).sha


def arditi():
    base = f"https://raw.githubusercontent.com/andyrdt/refusal_direction/{ARDITI_COMMIT}/dataset/splits"
    files = [fetch(f"{base}/{s}.json", DATA / "arditi" / "splits" / f"{s}.json") for s in ARDITI_SPLITS]
    for f in files:
        print(f"  {f.name}: {len(json.load(open(f)))} prompts")
    record("arditi", "github.com/andyrdt/refusal_direction", ARDITI_COMMIT, files)


def capabilities():
    from datasets import load_dataset
    out = DATA / "capabilities"
    out.mkdir(parents=True, exist_ok=True)
    rev = hf_revision("cais/mmlu", HF_REVISIONS["cais/mmlu"])
    ds = load_dataset("cais/mmlu", "all", split="test", revision=rev)
    with open(out / "mmlu_test.jsonl", "w") as f:
        for r in ds:
            f.write(json.dumps({"question": r["question"], "choices": r["choices"], "answer": r["answer"],
                                "subject": r["subject"]}) + "\n")
    record("mmlu", "hf:cais/mmlu (all, test)", rev, [out / "mmlu_test.jsonl"])
    rev = hf_revision("openai/gsm8k", HF_REVISIONS["openai/gsm8k"])
    ds = load_dataset("openai/gsm8k", "main", split="test", revision=rev)
    with open(out / "gsm8k_test.jsonl", "w") as f:
        for r in ds:
            f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
    record("gsm8k", "hf:openai/gsm8k (main, test)", rev, [out / "gsm8k_test.jsonl"])
    f = fetch(f"https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/{XSTEST_COMMIT}/xstest_prompts.csv",
              out / "xstest_prompts.csv")
    record("xstest_capabilities", "github.com/paul-rottger/exaggerated-safety", XSTEST_COMMIT, [f])
    print(f"  MMLU {sum(1 for _ in open(out / 'mmlu_test.jsonl'))} questions, GSM8K "
          f"{sum(1 for _ in open(out / 'gsm8k_test.jsonl'))} problems, XSTest")


def hf_snapshot(name, repo):
    from huggingface_hub import snapshot_download
    rev = hf_revision(repo, HF_REVISIONS[repo])
    d = Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=DATA / "published" / name))
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix in (".parquet", ".csv", ".tsv", ".json", ".jsonl")
             and ".cache" not in p.parts]
    record(name, f"hf:{repo}", rev, files)
    print(f"  {len(files)} data files")


def jailbreakbench():
    hf_snapshot("jailbreakbench", "JailbreakBench/JBB-Behaviors")
    base = f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{JBB_ARTIFACTS_COMMIT}/attack-artifacts"
    got = []
    for method in ("PAIR", "GCG", "JBC", "prompt_with_random_search"):
        for kind in ("black_box", "white_box", "manual", "transfer"):
            for target in ("llama-2-7b-chat-hf", "vicuna-13b-v1.5", "gpt-3.5-turbo-1106", "gpt-4-0125-preview"):
                try:
                    got.append(fetch(f"{base}/{method}/{kind}/{target}.json",
                                     DATA / "published" / "jailbreakbench" / f"{method}__{kind}__{target}.json"))
                except Exception:  # noqa: BLE001  (not every combination exists)
                    continue
    MANIFEST["jailbreakbench_artifacts"] = {"source": "github.com/JailbreakBench/artifacts",
                                            "revision": JBB_ARTIFACTS_COMMIT,
                                            "files": {str(p.relative_to(DATA)): sha256(p) for p in got}}
    print(f"  {len(got)} artifact files")


def published():
    for name, f in (("jailbreakbench", jailbreakbench),
                    ("wildjailbreak", lambda: hf_snapshot("wildjailbreak", "allenai/wildjailbreak")),
                    ("sorry_bench", lambda: hf_snapshot("sorry_bench", "sorry-bench/sorry-bench-202406")),
                    ("or_bench", lambda: hf_snapshot("or_bench", "bench-llm/or-bench")),
                    ("xstest", lambda: record("xstest", "github.com/paul-rottger/exaggerated-safety", XSTEST_COMMIT, [
                        fetch(f"https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/{XSTEST_COMMIT}/"
                              "xstest_prompts.csv", DATA / "published" / "xstest" / "xstest_prompts.csv")])),
                    ("harmbench", lambda: record("harmbench", "github.com/centerforaisafety/HarmBench", HARMBENCH_COMMIT, [
                        fetch(f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{HARMBENCH_COMMIT}/data/"
                              "behavior_datasets/harmbench_behaviors_text_all.csv",
                              DATA / "published" / "harmbench" / "harmbench_behaviors_text_all.csv")]))):
        attempt(name, f)


def attempt(name, f):
    print(f"\n=== {name}")
    try:
        f()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {str(e)[:300]}")
        traceback.print_exc(limit=1)
        return False


def main():
    ap = argparse.ArgumentParser(description="Download the published datasets at pinned revisions.")
    ap.add_argument("--only", default="arditi,capabilities,published")
    a = ap.parse_args()
    fm = DATA / "MANIFEST.json"
    if fm.exists():
        MANIFEST.update(json.load(open(fm)))
    todo = {x.strip() for x in a.only.split(",")}
    for name, f in (("arditi", arditi), ("capabilities", capabilities)):
        if name in todo:
            attempt(name, f)
    if "published" in todo:
        published()
    json.dump(MANIFEST, open(fm, "w"), indent=1)
    print(f"\nmanifest: {fm}")
    sys.exit(0)


if __name__ == "__main__":
    main()
