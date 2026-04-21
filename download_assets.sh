#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${REPO_DIR}/tools/download_bundle.py" assets "$@"
