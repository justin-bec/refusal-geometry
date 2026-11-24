"""Model specifications, loading, prompt formatting, and the batched operations every experiment uses.

An ``Engine`` holds one model, its tokenizer and its template, and the hooks registered on it. Every batched
operation takes a ``setup(rows, am)`` callback that configures the hooks for the rows of the current batch (each row
carries its own intervention); hooks are cleared after each pass.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .generation import align, cut, device_of, generate, pad_left, run_batches
from .hooks import StateReader
from .template import Template, find_template

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(os.environ.get("REFUSAL_MODELS", ROOT / "configs" / "models.yaml"))
LLAMA2_TEMPLATE = "[INST] {instruction} [/INST] "        # the hard-coded Llama-2 format of Arditi et al. (no system)
END_OF_TURN = ("<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>", "<|end|>")
PROBES = ("Hello.", "How do I bake bread?", "Write a short poem about the sea, please!", "x")


@dataclass
class ModelSpec:
    key: str
    id: str
    layer: int                           # main layer c: the state space is the template state at its input
    attention: str = "sdpa"              # "eager" or "sdpa" (see configs/models.yaml)
    template: str = "hf"                 # "hf" (tokenizer chat template) or "arditi" (Llama-2 format)
    refusal_words: list = field(default_factory=lambda: ["I"])   # first tokens of refusals (Arditi et al.)
    arditi_published: list = None        # (layer, position) published by Arditi et al., if any
    revision: str = None                 # Hugging Face commit of the weights (configs/pin_revisions.py)


def load_specs(path=None):
    import yaml
    raw = yaml.safe_load(open(path or CONFIG))
    return {k: ModelSpec(key=k, **v) for k, v in raw.items() if k != JUDGE_KEY}


JUDGE_KEY = "judge"


def judge_revision(model_id, path=None):
    """Revision of the judge pinned in configs/models.yaml (entry ``judge``), if its id matches."""
    import yaml
    j = (yaml.safe_load(open(path or CONFIG)) or {}).get(JUDGE_KEY) or {}
    return j.get("revision") if j.get("id") == model_id else None


def get_spec(name, path=None):
    """By key (``llama3``) or by Hugging Face id."""
    specs = load_specs(path)
    if name in specs:
        return specs[name]
    for s in specs.values():
        if s.id == name:
            return s
    raise KeyError(f"unknown model {name}; known: {', '.join(specs)}")


def dtype_of(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_hf(model_id, dtype=torch.bfloat16, device="cuda", attention=None, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kw = dict(dtype=dtype, device_map=device, revision=revision)
    if attention:
        kw["attn_implementation"] = attention
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def pad_id_of(tok):
    for t in (tok.pad_token_id, tok.unk_token_id, tok.eos_token_id):
        if t is not None:
            return t
    return 0


def end_of_turn(tok):
    """Token that ends an answer: <|eot_id|> for Llama-3 (not <|end_of_text|>, after which generation would go on
    with a new turn), <end_of_turn> for Gemma, <|im_end|> for Qwen; otherwise the tokenizer's eos."""
    for t in END_OF_TURN:
        try:
            i = tok.convert_tokens_to_ids(t)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
            return i
    return tok.eos_token_id


def folder_name(model_id):
    return model_id.replace("/", "__")


class Engine:
    def __init__(self, spec, dtype="bfloat16", device="cuda", mem_frac=0.9, seed=0, max_prompt_tokens=None,
                 model=None, tok=None):
        if torch.cuda.is_available() and mem_frac:
            torch.cuda.set_per_process_memory_fraction(mem_frac)
        self.spec = spec
        if model is None:
            model, tok = load_hf(spec.id, dtype_of(dtype), device, spec.attention, spec.revision)
        self.model, self.tok = model, tok
        self.dev = device_of(model)
        self.L = len(model.model.layers)
        self.d = model.config.hidden_size
        self.pad_id, self.eos_id = pad_id_of(tok), end_of_turn(tok)
        self.max_prompt_tokens = max_prompt_tokens
        self.P, ref = find_template([self._encode(t) for t in PROBES])
        self.template = Template(self.P, ref)
        self.template_tokens = [tok.decode([int(t)]) for t in ref]
        self.hooks = []
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    # ------------------------------------------------------------------ prompts
    def _encode(self, text):
        if self.spec.template == "arditi":
            s, add = LLAMA2_TEMPLATE.format(instruction=text), True
        else:
            s = self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                             tokenize=False)
            add = not (self.tok.bos_token and s.startswith(self.tok.bos_token))
        return self.tok(s, return_tensors="pt", add_special_tokens=add).input_ids[0]

    def encode(self, text):
        """Formatted prompt ids; prompts longer than ``max_prompt_tokens`` keep their beginning and the template."""
        x = self._encode(text)
        m = self.max_prompt_tokens
        if m and len(x) > m:
            x = torch.cat([x[:m - self.P], x[-self.P:]])
        return x

    def decode(self, toks):
        return self.tok.decode(toks, skip_special_tokens=True)

    def meta(self):
        return {"model": self.spec.id, "revision": self.spec.revision, "L": self.L, "d": self.d, "P": self.P,
                "layer": self.spec.layer,
                "template_tokens": self.template_tokens, "attention": self.spec.attention,
                "template_format": self.spec.template, "end_of_turn": self.tok.decode([self.eos_id])}

    # ------------------------------------------------------------------ hooks
    def add(self, hook):
        self.hooks.append(hook)
        return hook

    def clear(self):
        for h in self.hooks:
            h.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # ------------------------------------------------------------------ batched operations
    def sample(self, prompts, setup=None, temp=1.0, n_tokens=64, batch=32, clock=None, title="", checkpoint=None):
        """Generate one response per entry of ``prompts`` (list of id tensors). ``setup(rows, am)`` configures the
        hooks for the rows of a batch. Returns a list of (tokens without end of turn, tokens with it)."""
        zeros = torch.zeros

        def one(rows):
            ids, am, pos = pad_left([prompts[r] for r in rows], self.pad_id)
            try:
                if setup is not None:
                    setup(rows, am)
                with self.template.batch(zeros(len(rows), dtype=torch.long), ids):
                    tk = generate(self.model, ids, am, pos, torch.full((len(rows),), float(temp)), n_tokens, self.gen,
                                  self.eos_id)
            finally:
                self.clear()
            return [cut(tk[j], self.eos_id) for j in range(len(rows))]
        return run_batches(len(prompts), lambda r: len(prompts[r]), batch, one, clock, title, checkpoint)

    def logprobs(self, prompts, responses, setup=None, batch=16, full=False):
        """Teacher-forced reading of prompt + response. For each row: (sum of log-probabilities of the response
        tokens, number of tokens), or with ``full=True`` the (n, vocab) log-probabilities on CPU."""
        def one(rows):
            ids, am, pos, n_max, mask = align([list(prompts[r]) for r in rows], [list(responses[r]) for r in rows],
                                              self.pad_id)
            tail = torch.tensor([len(responses[r]) for r in rows], dtype=torch.long)
            try:
                if setup is not None:
                    setup(rows, am)
                with torch.no_grad(), self.template.batch(tail, ids):
                    lg = self.model(input_ids=ids.to(self.dev), attention_mask=am.to(self.dev),
                                    position_ids=pos.to(self.dev), use_cache=False,
                                    logits_to_keep=n_max + 1).logits[:, :n_max].float()
            finally:
                self.clear()
            lp = torch.log_softmax(lg, -1)
            if full:
                return [lp[j, n_max - len(responses[r]):].cpu() for j, r in enumerate(rows)]
            tgt = ids[:, ids.shape[1] - n_max:].to(self.dev)
            tl = lp.gather(-1, tgt[..., None]).squeeze(-1) * mask.to(self.dev)
            return [(float(tl[j].sum()), int(mask[j].sum())) for j in range(len(rows))]
        return run_batches(len(prompts), lambda r: len(prompts[r]) + len(responses[r]), batch, one)

    def kl_per_token(self, prompts, responses, setup, batch=16):
        """Mean over response tokens of KL(intact || intervened), the response being read back (teacher forcing)
        in both models. ``setup`` configures the intervention of each row."""
        def one(rows):
            ids, am, pos, n_max, mask = align([list(prompts[r]) for r in rows], [list(responses[r]) for r in rows],
                                              self.pad_id)
            tail = torch.tensor([len(responses[r]) for r in rows], dtype=torch.long)
            kw = dict(input_ids=ids.to(self.dev), attention_mask=am.to(self.dev), position_ids=pos.to(self.dev),
                      use_cache=False, logits_to_keep=n_max + 1)
            try:
                with torch.no_grad(), self.template.batch(tail, ids):
                    l0 = torch.log_softmax(self.model(**kw).logits[:, :n_max].float(), -1)
                    setup(rows, am)
                    l1 = torch.log_softmax(self.model(**kw).logits[:, :n_max].float(), -1)
            finally:
                self.clear()
            kt = (l0.exp() * (l0 - l1)).sum(-1)
            m = mask.to(self.dev)
            return ((kt * m).sum(1) / m.sum(1).clamp(min=1)).tolist()
        return run_batches(len(prompts), lambda r: len(prompts[r]) + len(responses[r]), batch, one)

    def template_states(self, prompts, layers, batch=16, clock=None, title=""):
        """Natural template states (no intervention) at the input of each layer of ``layers``:
        {layer: (N, P, d) float32}."""
        reader = StateReader(self.model, layers, self.template)

        def one(rows):
            ids, am, pos = pad_left([prompts[r] for r in rows], self.pad_id)
            with self.template.batch(torch.zeros(len(rows), dtype=torch.long), ids):
                S = reader.read(self.model, ids, am, pos)
            return [{l: S[l][j] for l in layers} for j in range(len(rows))]
        try:
            out = run_batches(len(prompts), lambda r: len(prompts[r]), batch, one, clock, title)
        finally:
            reader.remove()
        return {l: torch.stack([o[l] for o in out]) for l in layers}

    def mean_states(self, texts, batch=16, chunk=64):
        """Mean template state at the input of every layer over the prompts ``texts``: (L, P, d)."""
        pr = [self.encode(t) for t in texts]
        tot = torch.zeros(self.L, self.P, self.d)
        for k in range(0, len(pr), chunk):
            st = self.template_states(pr[k:k + chunk], list(range(self.L)), batch)
            tot += torch.stack([st[l].sum(0) for l in range(self.L)])
        return tot / len(pr)

    def first_token_logprobs(self, prompts, setup=None, batch=48, reduce=None):
        """Log-probabilities of the first generated token (prefill only). With ``reduce(lp, row)`` returns one value
        per row instead of the full vector."""
        def one(rows):
            ids, am, pos = pad_left([prompts[r] for r in rows], self.pad_id)
            try:
                if setup is not None:
                    setup(rows, am)
                with torch.no_grad(), self.template.batch(torch.zeros(len(rows), dtype=torch.long), ids):
                    lg = self.model(input_ids=ids.to(self.dev), attention_mask=am.to(self.dev),
                                    position_ids=pos.to(self.dev), use_cache=False, logits_to_keep=1).logits[:, -1]
            finally:
                self.clear()
            lp = torch.log_softmax(lg.float(), -1)
            if reduce is None:
                return [lp[j].cpu() for j in range(len(rows))]
            return [reduce(lp[j], r) for j, r in enumerate(rows)]
        return run_batches(len(prompts), lambda r: len(prompts[r]), batch, one)

    def free(self):
        self.remove_hooks()
        del self.model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
