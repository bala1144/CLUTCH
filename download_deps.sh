#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${REPO_DIR}/deps"
TMP_DIR="${DEPS_DIR}/.tmp"

T2M_DRIVE_ID="1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
GLOVE_URL="https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing"

mkdir -p "${DEPS_DIR}" "${TMP_DIR}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: missing required command '$1'"
        exit 1
    fi
}

run_gdown() {
    if command -v gdown >/dev/null 2>&1; then
        gdown "$@"
        return
    fi

    if command -v python >/dev/null 2>&1 && python -m gdown --help >/dev/null 2>&1; then
        python -m gdown "$@"
        return
    fi

    echo "ERROR: gdown is not available. Activate the CLUTCH environment and rerun."
    exit 1
}

download_t2m() {
    if [ -f "${DEPS_DIR}/t2m/t2m/text_mot_match/model/finest.tar" ]; then
        echo "[deps] t2m evaluator already present, skipping."
        return
    fi

    local archive="${TMP_DIR}/t2m.tar.gz"
    echo "[deps] Downloading TM2T evaluator bundle..."
    run_gdown "https://drive.google.com/uc?id=${T2M_DRIVE_ID}" -O "${archive}"
    echo "[deps] Extracting TM2T evaluator bundle..."
    tar xfz "${archive}" -C "${DEPS_DIR}"
    rm -f "${archive}"
}

download_glove() {
    if [ -f "${DEPS_DIR}/glove/our_vab_data.npy" ] && [ -f "${DEPS_DIR}/glove/our_vab_words.pkl" ] && [ -f "${DEPS_DIR}/glove/our_vab_idx.pkl" ]; then
        echo "[deps] glove assets already present, skipping."
        return
    fi

    need_cmd unzip
    local archive="${TMP_DIR}/glove.zip"
    echo "[deps] Downloading GloVe metadata bundle..."
    run_gdown --fuzzy "${GLOVE_URL}" -O "${archive}"
    echo "[deps] Extracting GloVe metadata bundle..."
    rm -rf "${DEPS_DIR}/glove"
    unzip -oq "${archive}" -d "${DEPS_DIR}"
    rm -f "${archive}"
}

prepare_placeholders() {
    mkdir -p "${DEPS_DIR}/mGPT_instructions" "${DEPS_DIR}/transforms" "${DEPS_DIR}/smpl"
}

download_t2m
download_glove
prepare_placeholders

cat <<EOF

Dependency bootstrap complete.

Downloaded:
- deps/t2m
- deps/glove

Prepared placeholder directories:
- deps/mGPT_instructions
- deps/transforms
- deps/smpl

Quantitative evaluation uses deps/t2m.
EgoVid5M dataloading uses deps/glove.
EOF
