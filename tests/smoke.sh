#!/usr/bin/env bash
# Runs every experiment on the tiny random models of tests/make_tiny.py, on CPU, with the fake judge.
# Checks that the code runs end to end; measures nothing. Needs REFUSAL_MODELS, REFUSAL_DATA, REFUSAL_RUNS.
set -euo pipefail
export PYTHONHASHSEED=0
C="--judge fake --dtype float32 --device cpu --mem_frac 0"
ALL="--models llama3,llama2,qwen,gemma"
cd "$(dirname "$0")/.."
python experiments/01_bottleneck.py $ALL $C --n_pairs 6 --n_layers 3 --tokens 6
python experiments/02_directions.py $ALL $C --n_train 40 --n_means 16 --k_border 3 --rounds 2 --k_grad 4 --n_null 3 --tokens 6
python experiments/03_arditi_selection.py $ALL $C --n_val 6
python experiments/04_ablate_everywhere.py $ALL $C --n_test 8 --n_kl 4 --tokens 6
python experiments/05_one_layer.py $ALL $C --n_test 6 --n_kl 4 --tokens 6 --n_mmlu 4 --n_gsm8k 2 --tokens_gsm8k 8
python experiments/06_one_vector.py $C --n_test 6 --n_kl 4 --n_harmless 4 --tokens 6 --n_mmlu 4 --n_gsm8k 2 --tokens_gsm8k 8
python experiments/07_thresholds.py $C --n_pairs 4 --n_alone 4 --k_nat 2 --k 2 --rounds 2 --remeasures 2 --k_fine 2 --rounds_fine 2 --tokens 6
python experiments/08_slopes.py $C --per_group 3 --k_grad 4 --tokens 6
python experiments/09_h0.py $C --n_prompts 5 --k 3 --tokens 6 --n_sim 20
python experiments/10_gains.py $C --n_prompts 4 --k 2 --rounds 2 --tokens 6
python experiments/11_threshold_analysis.py --n_boot 50
python experiments/12_width.py
export REFUSAL_DIRECTIONS="${REFUSAL_RUNS}/../directions"
python tools/export_directions.py
P="--directions published"
python experiments/03_arditi_selection.py --models llama2 $C $P --n_val 6 --n_means 16 --out "${REFUSAL_RUNS}/published_03"
python experiments/07_thresholds.py $C $P --n_pairs 3 --n_alone 3 --k_nat 2 --k 2 --rounds 2 --tokens 6 --out "${REFUSAL_RUNS}/published_07"
echo "smoke test: all scripts ran"
