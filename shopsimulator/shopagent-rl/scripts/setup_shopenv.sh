#!/usr/bin/env bash
# Build the ShopSimulator *service* environment (old stack) that runs pack_api.py @ :5000.
#
# This is INTENTIONALLY a separate env from `shopsimulator` (the training env):
#   shopenv       = Python 3.9 + 2022-era pins — runs the CPU-only Flask env service.
#                   (No ROCm/GPU here by design: gym 0.24 / flask 2.1 / pyserini 0.17 /
#                    spacy 3.4 are all CPU. The service never touches the GPU.)
#   shopsimulator = Python 3.12 + torch2.10/rocm + vLLM-rocm + veRL — runs SFT/GRPO on
#                   the AMD GPU (cloned from opd-rocm). Merging the two would break both.
#
# Fixes vs. the upstream requirements.txt (validated on this box):
#   * Flask 2.1.2 needs the OLD werkzeug/jinja2/markupsafe (reqs don't pin them; pip
#     otherwise pulls werkzeug 3.x which removes url_quote and breaks flask import).
#   * spacy bumped 3.3.0 -> 3.4.4: spacy 3.3 forces pydantic<1.9 -> pydantic 1.8.2,
#     which crashes on Python 3.9.25 (TypeError: issubclass()). 3.4.4 + pydantic 1.10.13
#     fixes it.
#   * spacy model download needs pip trusted-host for github (mihomo proxy MITMs TLS).
#
# Data + index are built separately in the upstream tree (00_start_env.sh preflight).
set -euo pipefail

CONDA=/workspace/persistent/miniconda3/bin/conda
ENV_DIR=/workspace/persistent/miniconda3/envs/shopenv
PIP="$ENV_DIR/bin/pip"
PY="$ENV_DIR/bin/python"

if [ -x "$PY" ]; then
  echo "[1/4] shopenv already exists, skipping create"
else
  echo "[1/4] conda create -n shopenv python=3.9"
  $CONDA create -n shopenv python=3.9 -y
fi

echo "[2/4] conda install openjdk + faiss-cpu (conda-forge)"
$CONDA install -n shopenv -c conda-forge openjdk=21 faiss-cpu -y

echo "[3/4] pip install old-stack deps (flask-2.1-compatible + spacy 3.4)"
$PIP install --no-input \
  gym==0.24.0 flask==2.1.2 "Werkzeug==2.0.3" "Jinja2==3.0.3" "itsdangerous==2.0.1" "MarkupSafe==2.0.1" \
  pyserini==0.17.0 spacy==3.4.4 "pydantic==1.10.13" \
  "numpy==1.22.4" "pandas==1.4.2" scikit_learn==1.1.1 thefuzz==0.19.0 \
  rank_bm25==0.2.2 rich==12.4.4 beautifulsoup4==4.11.1 "requests==2.27.1" \
  "PyYAML>=6.0" "tqdm>=4.64" torch==1.11.0

echo "[4/4] spacy model zh_core_web_sm (required at runtime by goal.py)"
$PIP config set global.trusted-host "github.com objects.githubusercontent.com codeload.github.com raw.githubusercontent.com github-releases.githubusercontent.com"
$PY -m spacy download zh_core_web_sm

echo "shopenv ready: $ENV_DIR"
