"""Prompt sets: the splits of Arditi et al., published jailbreak / benchmark prompts, capability benchmarks.

Nothing here generates or optimizes an attack: every prompt is read from a published dataset (see data/download.py,
which pins the revisions and records them in data/MANIFEST.json).
"""
import csv
import difflib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .models import ROOT

DATA = Path(os.environ.get("REFUSAL_DATA", ROOT / "data"))
ARDITI = DATA / "arditi" / "splits"
PUBLISHED = DATA / "published"
CAPABILITIES = DATA / "capabilities"


# ---------------------------------------------------------------------- Arditi et al. splits
def read_split(name, root=ARDITI):
    f = Path(root) / f"{name}.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run `python data/download.py` first")
    return [x if isinstance(x, str) else x.get("instruction", x.get("prompt")) for x in json.load(open(f))]


def arditi_sets(seed=0, n_test=128, n_kl=64, n_train=300, n_means=128, n_val=32, root=ARDITI):
    """The prompt sets used by every experiment on the four models (same seed -> same prompts everywhere):
      test_harmful   n_test harmful TEST prompts (never seen by any direction): ablations are measured on them;
      test_harmless  n_kl harmless TEST prompts: KL divergence of the interventions;
      train          up to n_train harmful prompts of train + val (minus test): boundary points and normals;
      means_harmful  the n_means first of them, means_harmless n_means harmless train prompts: mu+ and mu-;
      val_harmful, val_harmless   n_val validation prompts each, for the selection rule of Arditi et al."""
    r = random.Random(seed)
    th = read_split("harmful_test", root)
    r.shuffle(th)
    th = th[:n_test]
    tb = read_split("harmless_test", root)
    r.shuffle(tb)
    tb = tb[:n_kl]
    seen = set(th) | set(tb)
    train = [x for x in dict.fromkeys(read_split("harmful_train", root) + read_split("harmful_val", root))
             if x not in seen]
    r.shuffle(train)
    mh, train = train[:n_means], train[:n_train]
    mb = [x for x in read_split("harmless_train", root) if x not in seen]
    r.shuffle(mb)
    mb = mb[:n_means]
    rv = random.Random(seed + 7)
    vh = read_split("harmful_val", root)
    rv.shuffle(vh)
    vb = read_split("harmless_val", root)
    rv.shuffle(vb)
    return {"test_harmful": th, "test_harmless": tb, "train": train, "means_harmful": mh, "means_harmless": mb,
            "val_harmful": vh[:n_val], "val_harmless": vb[:n_val]}


# ---------------------------------------------------------------------- near-duplicate groups and folds
def content_words(texts, max_share=0.15):
    """Content words (those in at most max_share of the corpus, so that template words such as "write", "how" do
    not decide similarity) and their IDF weights."""
    n = len(texts)
    words = [set(re.findall(r"[a-z0-9]+", t.lower())) for t in texts]
    df = defaultdict(int)
    for w in words:
        for t in w:
            df[t] += 1
    cap = max(2.0, max_share * n)
    idf = {t: float(np.log(1.0 + n / c)) for t, c in df.items() if c <= cap}
    rare = [{t for t in w if t in idf} for w in words]
    return rare, [sum(idf[t] for t in r) for r in rare], idf


def duplicate_groups(texts, overlap=0.6, string_ratio=0.9):
    """Union-find of near duplicates: two texts are linked if the IDF-weighted overlap coefficient of their content
    words (at least two shared) reaches ``overlap``, or if their string similarity reaches ``string_ratio``.
    Returns one group id per text."""
    n = len(texts)
    rare, mass, idf = content_words(texts)
    low = [t.lower() for t in texts]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            common = rare[i] & rare[j]
            m = min(mass[i], mass[j])
            linked = len(common) >= 2 and m > 0 and sum(idf[t] for t in common) / m >= overlap
            if not linked:
                linked = difflib.SequenceMatcher(None, low[i], low[j]).ratio() >= string_ratio
            if linked:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(n)]


def two_folds(groups, seed=0):
    """Assign GROUPS (not prompts) to two folds of sizes as close as possible: two rephrasings of one request are
    always in the same fold. Returns 0 or 1 per element."""
    g = defaultdict(list)
    for i, k in enumerate(groups):
        g[k].append(i)
    blocks = list(g.values())
    random.Random(seed).shuffle(blocks)
    blocks.sort(key=len, reverse=True)
    folds = ([], [])
    for b in blocks:
        folds[0 if len(folds[0]) <= len(folds[1]) else 1].extend(b)
    out = [0] * len(groups)
    for i in folds[1]:
        out[i] = 1
    return out


# ---------------------------------------------------------------------- published prompts, by group
def _table(p):
    import pandas as pd
    p = Path(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".csv", ".tsv"):
        return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",", keep_default_na=False, on_bad_lines="skip")
    if p.suffix == ".jsonl":
        return pd.DataFrame([json.loads(x) for x in open(p) if x.strip()])
    if p.suffix == ".json":
        d = json.load(open(p))
        return pd.DataFrame(d if isinstance(d, list) else d.get("jailbreaks", d.get("data", [])))
    raise ValueError(f"unknown format: {p}")


def _col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _text(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return str(v[0]) if len(v) else ""
    return str(v)


def published_groups(root=PUBLISHED, log=print):
    """{group: [(prompt, base request or None), ...]}. A group = one dataset x one method or transformation:
      jbb_base_harmful, jbb_base_benign             JailbreakBench behaviors (Goal column)
      jbb_<method>_<target model>                    published JailbreakBench artifacts (base = goal)
      wj_vanilla_harmful, wj_adversarial_harmful, wj_vanilla_benign, wj_adversarial_benign      WildJailbreak
      sorry_base, sorry_<transformation>             SORRY-Bench (base = the untransformed question)
      orbench_<file>                                 OR-Bench (hard-1k, toxic; the 80k file is skipped)
      xstest_safe, xstest_unsafe                     XSTest
      harmbench_base                                 HarmBench standard behaviors without context
    A dataset that cannot be read (not downloaded, access not granted) is reported and skipped."""
    root = Path(root)
    G = {}

    def add(name, pairs):
        pairs = [(p.strip(), (b.strip() if b else None)) for p, b in pairs if isinstance(p, str) and p.strip()]
        if pairs:
            G.setdefault(name, []).extend(pairs)

    def attempt(name, f):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            log(f"  {name}: unreadable, skipped ({str(e)[:160]})")

    def _jbb():
        for p in sorted((root / "jailbreakbench").rglob("*")):
            if not p.is_file() or p.suffix not in (".csv", ".parquet", ".json", ".jsonl"):
                continue
            if "__" in p.name:
                J = json.load(open(p))
                meth, _, target = p.stem.split("__")
                add(f"jbb_{meth}_{target}", [(j.get("prompt"), j.get("goal")) for j in J.get("jailbreaks", [])
                                             if j.get("prompt")])
            else:
                df = _table(p)
                c = _col(df, ["Goal"])
                if c is not None:
                    add(f"jbb_base_{'benign' if 'benign' in p.name.lower() else 'harmful'}",
                        [(t, None) for t in df[c].astype(str)])

    def _wj():
        import pandas as pd
        for p in sorted(x for x in (root / "wildjailbreak").rglob("*") if x.is_file() and x.suffix in
                        (".tsv", ".csv", ".parquet")):
            if p.suffix == ".tsv":
                head = pd.read_csv(p, sep="\t", nrows=0)
                cols = [c for c in head.columns if c.lower() in ("vanilla", "adversarial", "data_type")]
                df = pd.read_csv(p, sep="\t", usecols=cols, keep_default_na=False, on_bad_lines="skip")
            else:
                df = _table(p)
            ct, cv, ca = _col(df, ["data_type"]), _col(df, ["vanilla"]), _col(df, ["adversarial"])
            if ct is None:
                continue
            for typ in ("vanilla_harmful", "vanilla_benign", "adversarial_harmful", "adversarial_benign"):
                s = df[df[ct] == typ]
                if not len(s):
                    continue
                if typ.startswith("vanilla") and cv:
                    add(f"wj_{typ}", [(t, None) for t in s[cv].astype(str)])
                elif typ.startswith("adversarial") and ca:
                    add(f"wj_{typ}", list(zip(s[ca].astype(str), s[cv].astype(str) if cv else [None] * len(s))))

    def _sorry():
        fs = sorted((root / "sorry_bench").rglob("question*.jsonl"))
        base = {}
        for p in fs:
            if p.stem == "question":
                for x in open(p):
                    if x.strip():
                        o = json.loads(x)
                        base[o.get("question_id")] = _text(o.get("turns", o.get("prompt", "")))
        for p in fs:
            mut = "base" if p.stem == "question" else p.stem.replace("question_", "")
            pairs = []
            for x in open(p):
                if x.strip():
                    o = json.loads(x)
                    pairs.append((_text(o.get("turns", o.get("prompt", ""))),
                                  None if mut == "base" else base.get(o.get("question_id"))))
            add(f"sorry_{mut}", pairs)

    def _or():
        for p in sorted((root / "or_bench").rglob("*")):
            if not p.is_file() or p.suffix not in (".csv", ".parquet") or "80k" in p.name:
                continue
            df = _table(p)
            c = _col(df, ["prompt"])
            if c:
                name = re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).replace("or_bench_", "").strip("_")
                add(f"orbench_{name}", [(t, None) for t in df[c].astype(str)])

    def _xs():
        df = _table(next((root / "xstest").rglob("*.csv")))
        c, cl = _col(df, ["prompt"]), _col(df, ["label"])
        for lab, name in (("safe", "xstest_safe"), ("unsafe", "xstest_unsafe")):
            add(name, [(t, None) for t in df[df[cl] == lab][c].astype(str)])

    def _hb():
        df = _table(next((root / "harmbench").rglob("*.csv")))
        c, cf, cc = _col(df, ["Behavior"]), _col(df, ["FunctionalCategory"]), _col(df, ["ContextString"])
        s = df
        if cf:
            s = s[s[cf] == "standard"]
        if cc:
            s = s[s[cc].astype(str).str.strip() == ""]
        add("harmbench_base", [(t, None) for t in s[c].astype(str)])

    for name, f in (("jailbreakbench", _jbb), ("wildjailbreak", _wj), ("sorry_bench", _sorry), ("or_bench", _or),
                    ("xstest", _xs), ("harmbench", _hb)):
        attempt(name, f)
    return G


BENIGN_GROUPS = ("jbb_base_benign", "wj_vanilla_benign", "wj_adversarial_benign", "xstest_safe", "orbench_hard_1k")


def family(group):
    """Dataset family of a group (jbb, wj, sorry, orbench, xstest, harmbench)."""
    return group.replace("base_of_", "").split("_")[0]


def stratified(indices, group_of, n, seed):
    """Draw up to n indices, taking one per group in turn (round robin over sorted groups)."""
    r = random.Random(seed)
    by = {}
    for i in indices:
        by.setdefault(group_of(i), []).append(i)
    for g in by:
        r.shuffle(by[g])
    out, k = [], 0
    while len(out) < min(n, len(indices)):
        for g in sorted(by):
            if k < len(by[g]) and len(out) < n:
                out.append(by[g][k])
        k += 1
    return out


# ---------------------------------------------------------------------- capability benchmarks
GSM8K_INSTRUCTION = "{q}\n\nSolve this step by step, then give the final answer as a plain number on the last line."


def _subsample(items, n, seed):
    rng = np.random.default_rng(seed)
    if n and len(items) > n:
        items = [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]
    return items


def read_mmlu(n, seed=0, path=CAPABILITIES / "mmlu_test.jsonl"):
    """[{question, choices, answer index}] (MMLU test split, Hugging Face format)."""
    items = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            items.append({"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"])})
    return _subsample(items, n, seed)


def read_gsm8k(n, seed=0, path=CAPABILITIES / "gsm8k_test.jsonl"):
    """[{question, target number, solution}]; the reference solution has its <<...>> annotations removed and ends
    with the answer."""
    items = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", r.get("answer", ""))
        if m and r.get("question"):
            target = m.group(1).replace(",", "")
            reasoning = re.sub(r"<<[^>]*>>", "", r["answer"].split("####")[0]).strip()
            items.append({"question": r["question"], "target": target,
                          "solution": f"{reasoning}\nThe final answer is {target}."})
    return _subsample(items, n, seed)


def read_xstest(path=CAPABILITIES / "xstest_prompts.csv"):
    """(prompts, labels "safe"/"unsafe")."""
    t, lab = [], []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        low = {str(k).lower(): v for k, v in r.items()}
        p = (low.get("prompt") or "").strip()
        if p:
            t.append(p)
            lab.append((low.get("label") or "").strip().lower())
    return t, lab


def last_number(text):
    m = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return m[-1].replace(",", "") if m else None
