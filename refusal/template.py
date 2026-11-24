"""The chat template: the P last tokens of every formatted prompt, identical for all prompts.

Hooks that read or modify the template state must find its P positions in the sequence the model is reading. With
LEFT padding, row b of a batch of length T ends with ``tail[b]`` tokens placed AFTER the template (0 when the sequence
stops at the template, as in a prefill; len(response) when a prompt and its response are read in one block, as for
REINFORCE, NLL or KL). The template therefore occupies positions T - tail[b] - P, ..., T - tail[b] - 1.

Every forward pass announces its ``tail`` with ``with template.batch(tail, ids):``, which also CHECKS that the tokens
found at these positions are the template tokens (immediate stop otherwise), and resets everything on exit.
"""
import torch


def find_template(id_lists):
    """Longest common suffix of encoded prompts: returns (P, reference ids of the template)."""
    P = 0
    shortest = min(len(x) for x in id_lists)
    while P < shortest - 1 and all(int(x[-P - 1]) == int(id_lists[0][-P - 1]) for x in id_lists):
        P += 1
    if P == 0:
        raise RuntimeError("the formatted prompts share no common suffix: no template found")
    return P, torch.as_tensor(id_lists[0][-P:]).clone()


class Template:
    def __init__(self, n_pos, ref_ids):
        self.n_pos = n_pos
        self.ref = torch.as_tensor(ref_ids, dtype=torch.long)
        self.tail = None

    class _Batch:
        def __init__(self, tpl, tail, ids):
            self.tpl, self.tail, self.ids = tpl, tail, ids

        def __enter__(self):
            tail = torch.as_tensor(self.tail, dtype=torch.long)
            B, T = self.ids.shape
            if tail.shape != (B,):
                raise RuntimeError(f"Template: tail of shape {tuple(tail.shape)} for a batch of {B}")
            start = T - tail - self.tpl.n_pos
            if bool((start < 0).any()):
                raise RuntimeError("Template: negative start position")
            idx = start[:, None] + torch.arange(self.tpl.n_pos)
            read = self.ids.gather(1, idx)
            if not bool((read == self.tpl.ref[None]).all()):
                b = int((read != self.tpl.ref[None]).any(1).nonzero()[0])
                raise RuntimeError(f"Template: row {b} carries {read[b].tolist()} instead of "
                                   f"{self.tpl.ref.tolist()} at the computed positions")
            self.tpl.tail = tail
            return self.tpl

        def __exit__(self, *exc):
            self.tpl.tail = None
            return False

    def batch(self, tail, ids):
        return Template._Batch(self, tail, ids.cpu())

    def indices(self, h):
        """(B, P) indices of the template positions in h (B, T, d)."""
        if self.tail is None:
            raise RuntimeError("Template: hook active outside a `with template.batch(...)` block")
        B, T = h.shape[0], h.shape[1]
        if self.tail.shape[0] != B:
            raise RuntimeError(f"Template: tail for {self.tail.shape[0]} rows, h has {B}")
        start = (T - self.tail - self.n_pos).to(h.device)
        return start[:, None] + torch.arange(self.n_pos, device=h.device)

    @staticmethod
    def read(h, idx):
        return h.gather(1, idx[..., None].expand(-1, -1, h.shape[2]))

    @staticmethod
    def write(h, idx, x):
        """Out of place and differentiable (with respect to h outside the template and to x)."""
        return h.scatter(1, idx[..., None].expand(-1, -1, h.shape[2]), x.to(h.dtype))
