"""Shared figures: harmful responses and refusals per condition (with Arditi et al. for comparison), and harmful
responses against the KL divergence. Every figure must be readable without the code: plain-word labels, legends
for every colour and marker, axes in words."""
import numpy as np

OKABE_ITO = ("#009E73", "#D55E00", "#0072B2", "#CC79A7", "#56B4E9", "#E69F00", "#332288", "#117733", "#88CCEE",
             "#999999", "#555555")
FIXED = {"no intervention": "#999999", "Arditi et al.": "#D55E00", "random direction": "#555555"}


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def colours(families):
    """A stable colour per family, Arditi et al. always vermillion, the no-intervention row grey."""
    out, k = {}, 0
    for f in families:
        if f in out:
            continue
        if f in FIXED:
            out[f] = FIXED[f]
        else:
            free = [c for c in OKABE_ITO if c not in FIXED.values()]
            out[f] = free[k % len(free)]
            k += 1
    return out


def ablation_figure(path, table, title, separator_after=None):
    """(a) bars of harmful responses (solid) and refusals (hatched) per condition, 95 % intervals;
    (b) harmful responses against the per-token KL on harmless prompts."""
    plt = mpl()
    from matplotlib.patches import Patch
    names = list(table)
    col = colours([table[n]["family"] for n in names])
    xs = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(max(14, 4 + 1.1 * len(names)), 6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    for dx, key, hatch, alpha in ((-0.2, "harmful", None, 0.9), (0.2, "refusal", "//", 0.55)):
        v = np.array([table[n][key] for n in names])
        ax.bar(xs + dx, v[:, 0], 0.4, color=[col[table[n]["family"]] for n in names], hatch=hatch,
               edgecolor="white", alpha=alpha)
        ax.errorbar(xs + dx, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="black",
                    capsize=3)
    if separator_after is not None:
        ax.axvline(separator_after + 0.5, color="black", ls=":", lw=1)
    ax.legend(handles=[Patch(facecolor="#777777", label="responses judged harmful (solid)"),
                       Patch(facecolor="#777777", alpha=0.55, hatch="//", edgecolor="white",
                             label="responses judged refusals (hatched)")]
              + [Patch(facecolor=c, label=f) for f, c in col.items()], fontsize=8, loc="upper left", ncol=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([table[n]["label"] for n in names], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.4)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("share of responses to harmful test prompts (95 % CI)")
    ax.set_title("(a) Harmful responses and refusals (coherent responses only)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax = axs[1]
    for n in names[1:]:
        ax.scatter(max(table[n]["kl"], 1e-4), table[n]["harmful"][0], s=70, color=col[table[n]["family"]],
                   label=table[n]["label"])
    ax.set_xscale("log")
    ax.set_xlabel("per-token KL on harmless prompts, intact model -> intervened model (log scale)")
    ax.set_ylabel("share of harmful responses")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) Harmful responses against the divergence they cost", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def spectrum_panel(ax, summ, title, n_vec, colour="#0072B2", label="eigenvalues of M"):
    """Eigenvalues of M normalised by lambda_1, with the 5-95 % bands of the two controls."""
    k = np.arange(1, n_vec + 1)
    w = np.array(summ["lambda"][:n_vec])
    for name, c, lab in (("rotation", "#999999", "control: normals replaced by random vectors of the same norm"),
                         ("permutation", "#E69F00", "control: state-normal pairing shuffled")):
        if f"null_{name}_q05" in summ:
            ax.fill_between(k, np.array(summ[f"null_{name}_q05"][:n_vec]) / w[0],
                            np.array(summ[f"null_{name}_q95"][:n_vec]) / w[0], color=c, alpha=0.3, label=lab)
    ax.plot(k, w / w[0], "-o", color=colour, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(k)
    ax.set_xlabel("rank of the eigenvalue")
    ax.set_ylabel("eigenvalue / first eigenvalue")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
