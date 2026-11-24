#!/usr/bin/env bash
# setup.sh -- creates .venv with the PyTorch build that matches the GPU of this machine, installs the requirements,
# and checks that PyTorch sees the GPU. Run once, from the repository root:
#     bash setup.sh            then    source .venv/bin/activate
#
# GPU detection:
#   NVIDIA (nvidia-smi present)     PyTorch from PyPI (CUDA build)
#   AMD (ROCm kernel driver, /sys/class/kfd): the gfx target is read from the driver, and PyTorch comes from the
#     TheRock nightly index of AMD for that target (https://rocm.nightlies.amd.com/v2/<family>/), which covers the
#     recent consumer GPUs and APUs (e.g. Strix Halo, gfx1151) that the pytorch.org ROCm builds do not.
# Overrides: PYTHON=python3.12 (interpreter), TORCH_INDEX=<url> (any PyTorch wheel index), TORCH_PRE=1 (pre-releases).
# The TheRock wheels are built for Python 3.12: on AMD the script looks for python3.12 first.
set -euo pipefail
cd "$(dirname "$0")"

die() { echo "setup.sh: $*" >&2; exit 1; }

gpu="none"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    gpu="nvidia"
elif [ -d "${KFD_TOPOLOGY:-/sys/class/kfd/kfd/topology}/nodes" ]; then
    gpu="amd"
fi

gfx=""
if [ "$gpu" = "amd" ]; then
    # gfx_target_version MMmmss (decimal) -> gfx<MM><mm in hex><ss in hex>, e.g. 110501 -> gfx1151
    for f in "${KFD_TOPOLOGY:-/sys/class/kfd/kfd/topology}"/nodes/*/properties; do
        v=$(awk '/gfx_target_version/ {print $2}' "$f")
        if [ -n "$v" ] && [ "$v" != "0" ]; then
            gfx=$(printf "gfx%d%x%x" $((v / 10000)) $(((v / 100) % 100)) $((v % 100)))
            break
        fi
    done
    [ -n "$gfx" ] || die "AMD driver found but no GPU node with a gfx target in /sys/class/kfd"
fi

if [ -z "${TORCH_INDEX:-}" ] && [ "$gpu" = "amd" ]; then
    case "$gfx" in
        gfx1151|gfx1150|gfx1152|gfx1153) family="$gfx" ;;
        gfx1200|gfx1201)                 family="gfx120X-all" ;;
        gfx1100|gfx1101|gfx1102|gfx1103) family="gfx110X-all" ;;
        gfx1030|gfx1031|gfx1032|gfx1034) family="gfx103X-dgpu" ;;
        gfx940|gfx941|gfx942)            family="gfx94X-dcgpu" ;;
        gfx950)                          family="gfx950-dcgpu" ;;
        gfx90a)                          family="gfx90a" ;;
        *) die "no known TheRock index for $gfx: set TORCH_INDEX to a PyTorch wheel index for your GPU" ;;
    esac
    TORCH_INDEX="https://rocm.nightlies.amd.com/v2/${family}/"
    TORCH_PRE=1
fi
[ "$gpu" = "none" ] && [ -z "${TORCH_INDEX:-}" ] && die "no NVIDIA or AMD GPU detected (set TORCH_INDEX to force)"

if [ -z "${PYTHON:-}" ]; then
    for p in python3.12 python3.11 python3.13 python3; do
        if command -v "$p" >/dev/null 2>&1; then PYTHON="$p"; break; fi
    done
fi
[ -n "${PYTHON:-}" ] || die "no python found"
pyver=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [ "$gpu" = "amd" ] && [ "$pyver" != "3.12" ]; then
    echo "setup.sh: WARNING: the TheRock wheels are built for Python 3.12, found $PYTHON = $pyver"
    echo "          (on Fedora: sudo dnf install python3.12, then rerun; or PYTHON=... bash setup.sh)"
fi

echo "GPU: $gpu ${gfx}; python: $PYTHON ($pyver); PyTorch index: ${TORCH_INDEX:-PyPI}"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
if [ -n "${TORCH_INDEX:-}" ]; then
    python -m pip install --index-url "$TORCH_INDEX" ${TORCH_PRE:+--pre} torch
else
    python -m pip install torch
fi
python -m pip install -r requirements.txt
python - <<'EOF'
import torch
ok = torch.cuda.is_available()
backend = f"ROCm/HIP {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
print(f"torch {torch.__version__}, {backend}, GPU visible: {ok}" + (f" ({torch.cuda.get_device_name(0)})" if ok else ""))
if not ok:
    raise SystemExit("setup.sh: PyTorch does not see the GPU")
EOF
echo "done: source .venv/bin/activate"
