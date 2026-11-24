# refusal-geometry

Code of the article **« Géométrie de la région de refus d'un modèle de langage : un axe, un seuil par prompt, une
région large »** (Justin Bec, 2026).

Aligned language models refuse harmful requests. Arditi et al. (2024) showed that removing one direction from
the residual stream suppresses refusal. This code studies the **region** of internal states where a model refuses.
It provides:

- **A removal direction e1**, built from the states and the normals to the border of the region. Removed at a
  single layer, it suppresses refusal at least as well as the direction of Arditi et al. removed at every layer.
  This holds on four models: Llama-3-8B-Instruct, Llama-2-7B-chat, Qwen2.5-7B-Instruct and Gemma-2-9B-it.
- **A single space where the decision is taken**: on Llama-3-8B, the template state of layer 14. In that space,
  the refusal of each prompt reads as a coordinate along an axis *m*, common to all prompts, minus a threshold
  specific to each prompt.
- **What a jailbreak moves**: published attacks move the threshold more than the state.

## Scope and ethics

- **Prompts.** Only published prompts are used: the splits of Arditi et al., JailbreakBench, WildJailbreak,
  SORRY-Bench, OR-Bench, XSTest, HarmBench, MMLU, GSM8K. No attack is generated or optimized. Nothing searches,
  by gradient or otherwise, for inputs or activations that make refusal fall. The margin to the border is never
  used as an objective.
- **What is not in this repository.**
  - No dataset: `data/download.py` fetches the datasets at pinned revisions and records what it fetched.
  - No generated response, result file or modified weight.
  - The repository contains code, and the directions e1, e1^concat and m (see *Published directions*).
- **Gated datasets.** WildJailbreak and SORRY-Bench are gated on Hugging Face: request access on their pages.

## Installation

```bash
bash setup.sh                      # creates .venv with the PyTorch build for this machine's GPU, checks it sees it
source .venv/bin/activate
hf auth login                      # models (Llama, Gemma) and gated datasets
python configs/pin_revisions.py --dry_run    # only to check: configs/models.yaml already pins the weights used
python data/download.py            # data/arditi, data/capabilities, data/published, data/MANIFEST.json
```

`setup.sh` detects the GPU:

- **NVIDIA**: PyTorch from PyPI (CUDA build).
- **AMD**: it reads the gfx target from the ROCm driver and takes PyTorch from AMD's TheRock index for that target.
  This index also covers recent consumer GPUs and APUs (e.g. Strix Halo, gfx1151) that the pytorch.org ROCm
  builds do not. Its wheels are built for Python 3.12.

`TORCH_INDEX=<url> bash setup.sh` forces another PyTorch build. Do not install PyTorch through
`requirements.txt`: it deliberately leaves PyTorch out.

The code itself runs unchanged on both: it only uses `torch.cuda`, which PyTorch maps to CUDA and to ROCm.

**Memory.** Each script holds the studied model and the WildGuard judge together in bfloat16: about 31 GB for
Gemma-2-9B, 30 GB for the others. Activation memory comes on top, especially for the REINFORCE backpropagation.
A batch that does not fit in memory is cut in two and retried, so smaller GPUs only run slower.

`--mem_frac` caps the share of GPU memory each process takes (default 0.9).

## Published directions

`directions/<model>/directions.safetensors` holds, for each of the four models, unit float32 vectors:

| tensor | shape | what it is |
|---|---|---|
| `e1` | (d,) | first eigenvector of the per-position matrix M at vertex 0. It is removed from one layer (all positions) or from every layer, like the direction of Arditi et al. |
| `e1_concat` | (P, d) | first eigenvector of the concatenated M at vertex 0, a unit vector of R^{P d}. It is removed from the template state of the main layer, at prefill. |
| `m` | (P, d) | normalized mean of the unit normals to the border (Llama-3-8B-Instruct only, the model where m was tested as an axis). |

`directions/<model>/metadata.json` gives:

- the model and the **revision of its weights**;
- the main layer, P and the template tokens;
- the sign conventions;
- the number of border prompts and the spectrum statistics;
- the commit of the code that produced them, and a hash.

The directions only make sense for these weights. `configs/models.yaml` pins the same revisions
(`python configs/pin_revisions.py` writes them from the local Hugging Face cache).

**Using them.** Pass `--directions published` to start from them instead of running `02_directions.py`. This
works for 03, 05 and 07–12. Scripts 04 and 06 also need the directions at vertex μ⁻, so they need
`02_directions.py`. The directions are produced by `tools/export_directions.py` from the outputs of
`02_directions.py`.

## Running the experiments

Run the scripts from the repository root, in this order:

```bash
python experiments/01_bottleneck.py
```

Each script:

- writes to its own folder, `runs/<script>/<model>/`;
- keeps its own journal there (`journal.log`), with the time elapsed since launch;
- prints where its figures are.

Every phase is saved, so the same command resumes after an interruption. `--redo <phase>` forces a phase to be
recomputed.

| script | what it measures | article | GPU time (order of magnitude, per model) |
|---|---|---|---|
| `01_bottleneck.py` | one-layer transplantation of the template state: sufficiency, necessity; main layer *c* | models and protocol; a fixed space (bottleneck) | 1–2 h |
| `02_directions.py` | means μ±, boundary points, normals (estimated twice), spectrum of *M*, e1, e1^concat, *m* | judged refusal and its gradient; boundary points; the axis *m*; the direction e1; spectrum (one vector) | 6–10 h |
| `03_arditi_selection.py` | the direction of Arditi et al., selected by their rule | models and protocol | 15 min |
| `04_ablate_everywhere.py` | e1 and Arditi et al. removed everywhere, vertex 0 and vertex μ⁻ | removing e1 everywhere | 1 h |
| `05_one_layer.py` | e1 at one layer: types of positions, layer sweep, capabilities | one layer, all positions | 3 h |
| `06_one_vector.py` | e1^concat removed from / added to the template state of one layer; capabilities | one vector | 3 h |
| `07_thresholds.py` | thresholds along *m* of ~1 300 published prompts; pairs (base, attack); remeasures | (H0) along *m*; a threshold per prompt | 20 h, + 25 h per remeasure |
| `08_slopes.py` | normals at ~600 anchors (two estimates each) | the slope of the border | 15 h |
| `09_h0.py` | local angle condition (CPU); refusal curves around the threshold (GPU) | (H0) along *m* | 6 h |
| `10_gains.py` | margin gain of added directions | adding and removing: (H1) | 4 h |
| `11_threshold_analysis.py` | AUCs of position, threshold, margin; decomposition of what an attack moves (CPU) | a threshold per prompt | seconds |
| `12_width.py` | width criterion on the slopes (CPU) | the width: (H2) | seconds |

**Which models each script runs on.**

- Scripts 01–05 run on the four models by default (`--models llama3,llama2,qwen,gemma`).
- Scripts 06–12 run on Llama-3-8B-Instruct, the model with a bottleneck.
- Models and their main layer, attention implementation and chat format are in `configs/models.yaml`.

**Dependencies.**

- 03 → needs 02.
- 04, 05, 06 → need 02 and 03.
- 07 → needs 02.
- 08, 10 → need 07.
- 09 → needs 07 and 08.
- 11 → needs 07.
- 12 → needs 08.

Script 07 is long. Run it first with `--remeasures 0` (the default), and add `--remeasures 2` later: the same
command reuses what is done.

## Reproducibility

Scripts mix all conditions and prompts in the same batches, and responses are sampled at temperature 1. Results
therefore reproduce **within run-to-run variability, not bit for bit**. In the article, two independent runs of
the same ablation differ by about two points.

Seeds are fixed (`--seed`). They select the same prompts in every script.

## Code

```
refusal/            the library
  models.py         model specs, loading, prompt formatting, batched sampling / reading / KL
  template.py       the P template positions, located and checked in every batch
  hooks.py          interventions set row by row: transplantation, injection, ablations (diagonal, coupled,
                    affine, by type of position), addition, gradient capture
  generation.py     batching (sorted by length, halved when out of memory, checkpointed), sampling loop
  judge.py          WildGuard as a continuous judge; coherence of a response
  reinforce.py      gradient of the judged refusal (REINFORCE, leave-one-out baseline, independent estimates)
  spectrum.py       low-rank eigen-decomposition of M, rotation and permutation controls
  dichotomy.py      thresholds along an axis for many (state, context) keys at once
  evaluate.py       ablation conditions (harmful, refusal, incoherent, KL), capabilities (MMLU, GSM8K, XSTest)
  data.py           Arditi et al. splits, published prompt groups, folds, capability benchmarks
  stats.py, plots.py, log.py, cli.py
experiments/        one script per result, numbered in run order
directions/         the published directions e1, e1^concat, m (safetensors) and their metadata
tools/              export_directions.py (writes directions/ from runs/02_directions)
setup.sh            creates the virtual environment for the GPU of the machine
configs/            models.yaml (the four models and the judge, with pinned weight revisions), pin_revisions.py
data/download.py    the published datasets at pinned revisions
tests/              smoke tests on tiny random models (CPU)
```

## Smoke test (CPU, a few minutes, no download)

The smoke test runs every script on tiny random models, with `--judge fake`. The fake judge gives random verdicts:
it checks that the code runs, it never measures anything.

```bash
python tests/make_tiny.py --out /tmp/tiny        # add --arditi_splits <path> to use the real splits
export REFUSAL_MODELS=/tmp/tiny/models.yaml REFUSAL_DATA=/tmp/tiny/data REFUSAL_RUNS=/tmp/tiny/runs
bash tests/smoke.sh
python -m pytest tests/test_hooks.py -q
```

## Citation

See `CITATION.cff`. This work builds on Arditi et al., *Refusal in language models is mediated by a single
direction* (NeurIPS 2024), whose dataset splits it uses (github.com/andyrdt/refusal_direction).

## License

MIT (see `LICENSE`).
