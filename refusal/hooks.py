"""Interventions on the residual stream, as forward hooks whose parameters are set PER ROW of a batch.

Every row of a batch carries its own intervention, so that all conditions and all prompts can be mixed in the same
batches. Conventions:
  - "input of layer l" is the input of decoder block l (``hidden_states[l]`` in transformers);
  - an ablation "at layer l" acts on the input of block l AND on the outputs of its attention and of its MLP, which
    are added to the residual stream (the protocol of Arditi et al.);
  - template-state hooks (transplantation, injection, coupled ablation, gradient capture) act at PREFILL only: during
    decoding the template positions are in the KV cache. When a prompt and its response are read in one block
    (REINFORCE, NLL, KL), they touch the template positions only, which reproduces exactly the generation-time
    computation;
  - diagonal ablations may act on generated tokens too (protocol of Arditi et al.), depending on the row's flags.
Every hook class has ``clear()`` (back to identity) and ``remove()``.
"""
import torch

from .template import Template


def pre_hook(fn):
    """Block pre-hook that works whether hidden_states is passed positionally or as a keyword."""
    def hook(mod, args, kwargs):
        if args:
            return (fn(args[0]),) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = fn(kwargs["hidden_states"])
        return args, kwargs
    return hook


def layers_of(model):
    return model.model.layers


class Transplant:
    """Transplantation of dose a toward mu: at the template positions, at layers 0..layer (input of each block),
    x <- (1 - a) x + a mu_l. Not above ``layer``: the operation has Jacobian (1 - a), so applying it higher up would
    multiply the gradient at ``layer`` by (1 - a)^(L - layer)."""

    def __init__(self, model, layer, template):
        self.tpl, self.a, self.mu = template, None, None
        self.handles = [layers_of(model)[l].register_forward_pre_hook(pre_hook(self._f(l)), with_kwargs=True)
                        for l in range(layer + 1)]

    def set(self, a, mu):
        """a: (B,) doses; mu: (L, P, d) target states."""
        self.a, self.mu = a, mu

    def clear(self):
        self.a = self.mu = None

    def _f(self, l):
        def g(h):
            if self.a is None or h.shape[1] <= 1:
                return h
            idx = self.tpl.indices(h)
            a = self.a.to(h.device, torch.float32)[:, None, None]
            x = Template.read(h, idx).float()
            mu = self.mu[l].to(h.device, torch.float32)[None]
            return Template.write(h, idx, (1 - a) * x + a * mu)
        return g

    def remove(self):
        for h in self.handles:
            h.remove()


class Inject:
    """Replaces the template state at the input of ``layer`` by a given state, row by row (prefill only). Rows whose
    ``active`` flag is False are left exactly as they are."""

    def __init__(self, model, layer, template):
        self.tpl, self.states, self.active = template, None, None
        self.handle = layers_of(model)[layer].register_forward_pre_hook(pre_hook(self._f), with_kwargs=True)

    def set(self, states, active):
        """states: (B, P, d); active: (B,) bool."""
        self.states, self.active = states, active

    def clear(self):
        self.states = self.active = None

    def _f(self, h):
        if self.states is None or h.shape[1] <= 1:
            return h
        idx = self.tpl.indices(h)
        x = Template.read(h, idx).float()
        new = torch.where(self.active.to(h.device)[:, None, None], self.states.to(h.device, torch.float32), x)
        return Template.write(h, idx, new)

    def remove(self):
        self.handle.remove()


class Ablation:
    """Directional ablations, set row by row.

    DIAGONAL: a unit direction U[b, l] (zero where inactive) removed at layer l, h <- h - (<u, h> - gamma) u, where
        gamma = 0 (vertex 0, Arditi et al.) or, on block inputs only, the per-layer offset ``offset[b, l]`` (affine
        ablation at vertex mu-, Marshall et al.: the coordinate along u is brought back to its mean over harmless
        prompts; the attention and MLP outputs always get vertex 0).
        Restricted per row to position TYPES, ``flags[b] = (instruction, template, generated)``: instruction = real
        positions before the template (including the begin token and the user header), template = the P last
        prompt positions, generated = the response tokens (sampled, or read back). Optionally the first real token
        of the prompt is spared (``skip_first[b]``; it often carries massive activations).
    COUPLED: a direction V[b, l] of R^{P d} removed from the template state at layer l, the coefficient being summed
        over the P positions (prefill only).
    ``start[b]`` = index of the first real token (left padding), required with flags or skip_first."""

    def __init__(self, model, template):
        self.tpl = template
        self.L = len(layers_of(model))
        self.clear()
        self.handles = []
        for l, blk in enumerate(layers_of(model)):
            self.handles.append(blk.register_forward_pre_hook(pre_hook(self._entry(l)), with_kwargs=True))
            self.handles.append(blk.self_attn.register_forward_hook(self._out(l)))
            self.handles.append(blk.mlp.register_forward_hook(self._out(l)))

    def set(self, U=None, V=None, offset=None, flags=None, skip_first=None, start=None):
        """U: (B, L, d); V: (B, L, P, d); offset: (B, L); flags: (B, 3) bool; skip_first: (B,) bool; start: (B,)."""
        self.U, self.V, self.offset, self.flags, self.skip, self.start = U, V, offset, flags, skip_first, start
        self.on_diag = [U is not None and bool(U[:, l].abs().amax() > 0) for l in range(self.L)]
        self.on_coup = [V is not None and bool(V[:, l].abs().amax() > 0) for l in range(self.L)]

    def clear(self):
        self.U = self.V = self.offset = self.flags = self.skip = self.start = None
        self.on_diag = self.on_coup = [False] * self.L

    def _mask(self, h):
        """(B, T) bool: positions of each row where the diagonal ablation applies, or None (everywhere)."""
        if self.flags is None and self.skip is None:
            return None
        B, T = h.shape[0], h.shape[1]
        if T == 1:                                            # decoding step: the token being generated
            if self.flags is None:
                return None
            return self.flags[:, 2].to(h.device)[:, None]
        idx = torch.arange(T, device=h.device)[None]
        start = self.start.to(h.device)[:, None]
        m = torch.ones(B, T, dtype=torch.bool, device=h.device)
        if self.flags is not None:
            tail = self.tpl.tail.to(h.device)[:, None]
            t0 = T - tail - self.tpl.n_pos
            f = self.flags.to(h.device)
            instr = (idx >= start) & (idx < t0)
            tpl = (idx >= t0) & (idx < T - tail)
            gen = idx >= T - tail
            m = (instr & f[:, :1]) | (tpl & f[:, 1:2]) | (gen & f[:, 2:3])
        if self.skip is not None:
            m = m & ~((idx == start) & self.skip.to(h.device)[:, None])
        return m

    def _diag(self, l, h, entry):
        u = self.U[:, l].to(h.device, torch.float32)               # (B, d)
        x = h.float()
        c = torch.einsum("btd,bd->bt", x, u)
        if entry and self.offset is not None:
            c = c - self.offset[:, l].to(h.device, torch.float32)[:, None]
        delta = c[..., None] * u[:, None, :]
        m = self._mask(h)
        if m is not None:
            delta = delta * m[..., None].to(delta.dtype)
        return (x - delta).to(h.dtype)

    def _coupled(self, l, h):
        if h.shape[1] <= 1:
            return h
        idx = self.tpl.indices(h)
        v = self.V[:, l].to(h.device, torch.float32)               # (B, P, d)
        x = Template.read(h, idx).float()
        c = torch.einsum("bpd,bpd->b", x, v)
        return Template.write(h, idx, x - c[:, None, None] * v)

    def _entry(self, l):
        def f(h):
            if self.on_diag[l]:
                h = self._diag(l, h, True)
            if self.on_coup[l]:
                h = self._coupled(l, h)
            return h
        return f

    def _out(self, l):
        def hook(mod, inp, out):
            if not self.on_diag[l]:
                return out
            if isinstance(out, tuple):
                return (self._diag(l, out[0], False),) + tuple(out[1:])
            return self._diag(l, out, False)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class Addition:
    """Adds, at the input of layer ``layer[b]`` only, the vector ``vec[b]`` at every position of row b
    (activation addition, used by the selection rule of Arditi et al.)."""

    def __init__(self, model):
        self.layer, self.vec = None, None
        self.handles = [blk.register_forward_pre_hook(pre_hook(self._f(l)), with_kwargs=True)
                        for l, blk in enumerate(layers_of(model))]

    def set(self, layer, vec):
        """layer: (B,) long (-1: none); vec: (B, d)."""
        self.layer, self.vec = layer, vec

    def clear(self):
        self.layer = self.vec = None

    def _f(self, l):
        def f(h):
            if self.vec is None:
                return h
            m = (self.layer == l).to(h.device)
            if not bool(m.any()):
                return h
            return (h.float() + (self.vec.to(h.device, torch.float32) * m[:, None].float())[:, None, :]).to(h.dtype)
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


class GradCapture:
    """Template state at the input of ``layer`` made differentiable there: after backward, ``grad()`` returns
    d(objective)/d(template state), all downstream paths included. Register it AFTER any hook acting on the same
    layer (transplantation, injection) so that it sees the modified state."""

    def __init__(self, model, layer, template):
        self.tpl, self.active, self.h, self.idx = template, False, None, None
        self.handle = layers_of(model)[layer].register_forward_pre_hook(pre_hook(self._f), with_kwargs=True)

    def _f(self, h):
        if not self.active or h.shape[1] <= 1:
            return h
        h = h.detach().requires_grad_(True)
        self.h, self.idx = h, self.tpl.indices(h)
        return h

    def grad(self):
        g = Template.read(self.h.grad, self.idx).detach().float().cpu()
        self.clear()
        return g

    def clear(self):
        self.active, self.h, self.idx = False, None, None

    def remove(self):
        self.handle.remove()


class _Stop(Exception):
    pass


class StateReader:
    """Reads the template state at the input of each layer of ``layers`` during a prefill. With ``stop=True`` the
    forward pass is interrupted after the last of these layers (nothing above is computed)."""

    def __init__(self, model, layers, template, stop=True):
        self.tpl, self.layers, self.stop = template, sorted(layers), stop
        self.states, self.active = {}, False
        self.handles = [layers_of(model)[l].register_forward_pre_hook(pre_hook(self._f(l)), with_kwargs=True)
                        for l in self.layers]

    def _f(self, l):
        def g(h):
            if self.active and h.shape[1] > 1:
                self.states[l] = Template.read(h, self.tpl.indices(h)).detach().float().cpu()
                if self.stop and l == self.layers[-1]:
                    raise _Stop
            return h
        return g

    def read(self, model, ids, am, pos):
        """{layer: (B, P, d)} for one batch (ids already inside a ``template.batch`` block)."""
        dev = next(model.parameters()).device
        self.states, self.active = {}, True
        try:
            with torch.no_grad():
                model(input_ids=ids.to(dev), attention_mask=am.to(dev), position_ids=pos.to(dev), use_cache=False,
                      logits_to_keep=1)
        except _Stop:
            pass
        finally:
            self.active = False
        return self.states

    def clear(self):
        self.active = False

    def remove(self):
        for h in self.handles:
            h.remove()
