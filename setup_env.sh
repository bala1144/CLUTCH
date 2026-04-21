#!/bin/bash
# CLUTCH environment setup script
# Creates a conda environment with all required dependencies.
# Tested on: Ubuntu 20.04/22.04, CUDA 12.1, Python 3.9

set -e  # exit on first error

ENV_NAME="clutch"
PYTHON_VERSION="3.9"

echo "============================================"
echo "  CLUTCH Environment Setup"
echo "============================================"

# ── 0. Resolve conda/mamba ───────────────────────
if command -v conda &> /dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_BIN="${CONDA_EXE}"
elif [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
    CONDA_BIN="${HOME}/miniforge3/bin/conda"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
    echo "ERROR: conda not found. Please install Miniconda, Miniforge, or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
CONDA_ROOT="$(cd "$(dirname "$(dirname "${CONDA_BIN}")")" && pwd)"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"

MAMBA_BIN=""
if command -v mamba &> /dev/null; then
    MAMBA_BIN="$(command -v mamba)"
elif [[ -n "${MAMBA_EXE:-}" && -x "${MAMBA_EXE}" ]]; then
    MAMBA_BIN="${MAMBA_EXE}"
fi

eval "$("${CONDA_BIN}" shell.bash hook)"

INSTALL_BIN="${CONDA_BIN}"
if [[ -n "${MAMBA_BIN}" ]]; then
    INSTALL_BIN="${MAMBA_BIN}"
fi

# ── 1. Create base conda environment ────────────
echo ""
echo "[1/4] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."

if [[ -d "${ENV_PREFIX}" ]]; then
    echo "      Environment '${ENV_NAME}' already exists. Reusing it."
else
    "${INSTALL_BIN}" create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# Activate the new environment
conda activate "${ENV_NAME}"

echo "      Environment '${ENV_NAME}' activated."

# ── 2. Install PyTorch with CUDA 12.1 ────────────
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "[2/4] Installing PyTorch 2.1 + CUDA 12.1..."
    "${CONDA_BIN}" install --solver classic -y -n "${ENV_NAME}" \
        pytorch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        pytorch-cuda=12.1 \
        -c pytorch -c nvidia
else
    echo "[2/4] Installing CPU-only PyTorch 2.1 with pip..."
    pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2
fi

# ── 3. Install pip packages ───────────────────────
echo ""
echo "[3/4] Installing Python packages..."

pip install \
    'numpy<1.24' \
    pytorch-lightning==2.4.0 \
    lightning==2.5.0 \
    transformers==4.36.2 \
    diffusers==0.32.2 \
    smplx==0.1.28 \
    wandb==0.19.6 \
    einops==0.8.0 \
    easydict==1.13 \
    h5py==3.8.0 \
    datasets==2.9.0 \
    evaluate==0.4.3 \
    bert-score==0.3.13 \
    ftfy==6.3.1 \
    chumpy==0.70 \
    imageio-ffmpeg==0.6.0 \
    fuzzywuzzy==0.18.0 \
    python-Levenshtein \
    colorlog \
    huggingface-hub==0.28.0 \
    gdown \
    accelerate \
    sentencepiece \
    spacy==3.7.5 \
    scipy \
    scikit-learn \
    seaborn \
    matplotlib \
    tqdm \
    rich \
    omegaconf==2.3.0 \
    trimesh==3.9.24 \
    pyrender==0.1.45 \
    open3d \
    PyOpenGL \
    opencv-python \
    Pillow

echo ""
echo "[4/4] Configuring environment variables..."
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${REPO_DIR}/assets"
mkdir -p "${ASSETS_DIR}"
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"

cat > "${CONDA_PREFIX}/etc/conda/activate.d/clutch_env.sh" <<EOF
export CLUTCH_ROOT="${REPO_DIR}"
export ASSETS_PATH="${ASSETS_DIR}"
export ASSESTS_PATH="${ASSETS_DIR}"
export PYOPENGL_PLATFORM=egl
export PYGLET_HEADLESS=1
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH}"
EOF

cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/clutch_env.sh" <<'EOF'
unset CLUTCH_ROOT
unset ASSETS_PATH
unset ASSESTS_PATH
unset PYOPENGL_PLATFORM
unset PYGLET_HEADLESS
EOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Download external dependencies into ./deps"
echo "     ./download_deps.sh"
echo "  2. Download repo assets into ./assets"
echo "     ./download_assets.sh --url <uploaded-assets-zip-url>"
echo "  3. Download pretrained checkpoints (see README.md)"
echo "  4. Set dataset and checkpoint paths in configs/"
echo ""
echo "Notes:"
echo "  - This setup is pip-first so text-to-motion inference works without host-specific conda solver behavior."
echo "  - Headless rendering prefers EGL and enables pyglet headless mode automatically."
