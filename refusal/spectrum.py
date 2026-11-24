"""The state--normal matrix M and its two null controls.

M = 1/2 sum_i (a_i g_i^T + g_i a_i^T), a_i = state - vertex, g_i = normal (gradient of the judged refusal at a
boundary point). By Ky Fan, its first eigenvector is the direction whose ablation lowers sum_i q_i the most, to
first order. M has rank <= 2N: it is never formed; a QR decomposition of [A, G] reduces the problem to 2N x 2N.

Controls:
  - permutation: each g is paired with the g of another sample (uniform permutation). Keeps the population part
    (mean state, mean normal) in expectation and destroys the covariance part;
  - rotation: each g is replaced by a random vector of the same norm. Destroys both; an eigenvalue above this
    control is a real direction.
"""
import numpy as np
import torch


def low_rank_eig(A, G, n_vec, n_perm=0, seed=0):
    """A, G: (D, N). Returns (eigenvalues sorted decreasingly, the n_vec first eigenvectors (n_vec, D),
    eigenvalues of the permutation control (n_perm, n_vec) or None)."""
    D, N = A.shape
    Q, R = torch.linalg.qr(torch.cat([A, G], 1).float(), mode="reduced")
    k = Q.shape[1]

    def kernel(Pm):
        S = torch.zeros(2 * N, 2 * N)
        S[:N, N:] = 0.5 * Pm.T
        S[N:, :N] = 0.5 * Pm
        K = R @ S @ R.T
        return 0.5 * (K + K.T)

    w, V = torch.linalg.eigh(kernel(torch.eye(N)))
    order = torch.argsort(w, descending=True)
    w, V = w[order], V[:, order]
    vecs = (Q @ V[:, :min(n_vec, k)]).T
    vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    null = None
    if n_perm:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_perm):
            Pm = torch.zeros(N, N)
            Pm[np.arange(N), rng.permutation(N)] = 1.0
            null.append(torch.linalg.eigvalsh(kernel(Pm)).flip(0)[:n_vec])
        null = torch.stack(null)
    return w, vecs, null


def rotation_null(A, G, n_vec, n, seed=0):
    """Eigenvalues (n, n_vec) when every g is replaced by a random vector of the same norm."""
    g = torch.Generator().manual_seed(seed)
    norms = G.norm(dim=0, keepdim=True)
    out = []
    for _ in range(n):
        Rn = torch.randn(G.shape, generator=g)
        Rn = Rn / Rn.norm(dim=0, keepdim=True).clamp(min=1e-12) * norms
        out.append(low_rank_eig(A, Rn, n_vec)[0][:n_vec])
    return torch.stack(out)


def summary(w, null_perm, null_rot, n_vec):
    """Numbers reported for a spectrum."""
    w = np.asarray(w, float)
    pos = w[w > 0].sum()
    r = {"lambda": w[:n_vec].tolist(), "lambda_min": float(w[-1]), "ratio_1_2": float(w[0] / w[1]),
         "share_lambda1": float(w[0] / pos) if pos > 0 else float("nan")}
    for name, null in (("rotation", null_rot), ("permutation", null_perm)):
        if null is not None:
            nu = np.asarray(null, float)
            r[f"lambda1_over_{name}"] = float(w[0] / nu[:, 0].mean())
            r[f"above_{name}"] = int((w[:n_vec] > nu.max()).sum())
            r[f"null_{name}_q05"] = np.quantile(nu, 0.05, 0).tolist()
            r[f"null_{name}_q95"] = np.quantile(nu, 0.95, 0).tolist()
    return r


def orient(v, ref):
    """Sign of v chosen so that <v, ref> >= 0."""
    return v if float(v.reshape(-1) @ ref.reshape(-1)) >= 0 else -v
