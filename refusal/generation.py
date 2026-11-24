"""Batching, generation and teacher-forced reading.

Rules followed everywhere:
  - work in large batches: all prompts and all conditions are mixed in the same batches, never prompt by prompt;
  - rows are sorted by length before forming batches (less padding);
  - a batch that does not fit in memory is cut in two and retried;
  - generation uses a hand-written loop with a KV cache and the Gumbel trick (no torch.multinomial, which crashes
    the GPU on a non-finite logit), so that no library warning is repeated at each call.
"""
import gc
import time
from pathlib import Path

import torch

OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def device_of(model):
    return next(model.parameters()).device


def pad_left(id_lists, pad_id):
    """Left padding (the last positions are aligned), explicit position ids. -> (ids, attention_mask, positions)."""
    T = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), T), pad_id, dtype=torch.long)
    am = torch.zeros((len(id_lists), T), dtype=torch.long)
    for j, x in enumerate(id_lists):
        x = torch.as_tensor(x, dtype=torch.long)
        ids[j, T - len(x):] = x
        am[j, T - len(x):] = 1
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    return ids, am, pos


def first_real(am):
    """(B,) index of the first real (non padding) token of each row."""
    return (am == 0).sum(1)


def sorted_batches(n, length, size):
    order = sorted(range(n), key=lambda i: -length(i))
    return [order[k:k + size] for k in range(0, n, size)]


def halving(fn, batch):
    """fn(batch) -> list of results; on out-of-memory, the batch is cut in two (recursively)."""
    try:
        return fn(batch)
    except OOM:
        if len(batch) == 1:
            raise
        gc.collect()
        torch.cuda.empty_cache()
        m = len(batch) // 2
        return halving(fn, batch[:m]) + halving(fn, batch[m:])


def run_batches(n, length, size, fn, clock=None, title="", checkpoint=None, every_s=600):
    """Apply ``fn`` (list of row indices -> list of results) to all n rows, in batches sorted by length.
    A progress line about every quarter. With ``checkpoint`` (a file path), partial results are saved every
    ``every_s`` seconds and reloaded by the same command after an interruption; the file is deleted at the end."""
    out = [None] * n
    if checkpoint is not None and Path(checkpoint).exists():
        Z = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if Z.get("n") == n:
            out = Z["out"]
            if clock:
                clock(f"    {title}: resumed, {sum(x is not None for x in out)}/{n} results reloaded")
    todo = [b for b in sorted_batches(n, length, size) if any(out[i] is None for i in b)]
    t0 = t_save = time.time()
    mark = max(1, len(todo) // 4)
    for k, b in enumerate(todo):
        for i, v in zip(b, halving(fn, b)):
            out[i] = v
        if checkpoint is not None and time.time() - t_save > every_s:
            torch.save({"n": n, "out": out}, checkpoint)
            t_save = time.time()
        if clock and title and ((k + 1) % mark == 0) and k + 1 < len(todo):
            el = time.time() - t0
            clock(f"    {title}: {k + 1}/{len(todo)} batches ({el / 60:.0f} min; about "
                  f"{el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min left, overestimated: long batches come first)")
    if checkpoint is not None and Path(checkpoint).exists():
        Path(checkpoint).unlink()
    return out


@torch.no_grad()
def generate(model, ids, am, pos, temps, n_new, gen, eos_id):
    """Sampling with a KV cache. temps (B,): temperature per row, <= 0 for greedy. Returns (B, n_new) tokens
    (positions after the end of turn are filled with eos_id)."""
    dev = device_of(model)
    ids, am, pos, temps = ids.to(dev), am.to(dev), pos.to(dev), temps.to(dev)
    B = ids.shape[0]

    def clean(lg):
        lg = lg.float()
        if not bool(torch.isfinite(lg).all()):
            lg = torch.nan_to_num(lg, nan=-1e4, posinf=1e4, neginf=-1e4)
        return lg

    o = model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    past, lg = o.past_key_values, clean(o.logits[:, -1])
    toks = torch.full((B, n_new), eos_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    greedy = temps <= 0
    t_eff = torch.where(greedy, torch.ones_like(temps), temps)[:, None]
    p = pos[:, -1:]
    for k in range(n_new):
        u = torch.rand(lg.shape, generator=gen, device=dev).clamp_(1e-10, 1 - 1e-7)
        t = torch.where(greedy, lg.argmax(-1), (lg / t_eff - torch.log(-torch.log(u))).argmax(-1))
        t = torch.where(done, torch.full_like(t, eos_id), t)
        toks[:, k] = t
        done |= t == eos_id
        if k == n_new - 1 or bool(done.all()):
            break
        am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=dev)], 1)
        p = p + 1
        o = model(input_ids=t[:, None], attention_mask=am, position_ids=p, past_key_values=past, use_cache=True)
        past, lg = o.past_key_values, clean(o.logits[:, -1])
    del past, o
    return toks.cpu()


def cut(toks, eos_id):
    """(tokens without the end of turn, tokens with it) of one generated row."""
    t = toks.tolist()
    n = t.index(eos_id) if eos_id in t else len(t)
    return t[:n], t[:min(n + 1, len(t))]


def align(prompts, responses, pad_id):
    """Prompt + response read in one block, left padded. Returns (ids, am, pos, n_max, mask) where the logits at
    the n_max + 1 last positions predict the response tokens and ``mask`` (B, n_max) marks the real ones."""
    seqs = [list(p) + list(r) for p, r in zip(prompts, responses)]
    ids, am, pos = pad_left(seqs, pad_id)
    n_max = max(1, max(len(r) for r in responses))
    mask = torch.zeros(len(responses), n_max, dtype=torch.bool)
    for b, r in enumerate(responses):
        if len(r):
            mask[b, n_max - len(r):] = True
    return ids, am, pos, n_max, mask


def trigram_diversity(toks):
    """Share of distinct token trigrams in a response (1 for short ones). Below 1/2: degenerate repetition."""
    if len(toks) < 3:
        return 1.0
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return len(set(tri)) / len(tri)
