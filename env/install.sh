#!/bin/bash
# =============================================================================
# install.sh -- one-off installation of the TERRA software environment
#
# Run once, from the repository root, on a node with internet access 
#     bash env/install.sh
#
# Steps
#   1. pull the PyTorch 2.6 / CUDA 12.4 image into a single .sif file
#   2. install the pinned Python packages (env/versions/requirements.lock,
#      including terra-st) with pip --user into $PYUSER, using the
#      container's Python
#   3. check that pip kept the image's torch and left no broken requirements
#   4. download TERRA-112M at the pinned revision and verify its checksum
#   5. record the installed versions in $SOFTWARE_DIR/logs/
# Safe to re-run: the image and model are skipped when present.
# =============================================================================
set -euo pipefail

source env/terra_env.sh

mkdir -p "$SOFTWARE_DIR"/{containers,pyuser,models,logs}
STAMP=$(date +%Y%m%d_%H%M)
LOG=$SOFTWARE_DIR/logs/install_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "0. Pre-flight"
echo "host: $(hostname)   log: $LOG"
command -v singularity >/dev/null || { echo "singularity not found"; exit 1; }
singularity --version
curl -sSf -o /dev/null --max-time 15 https://pypi.org/simple/ \
  || { echo "No internet access from this node"; exit 1; }

step "1. Container image"
if [[ -s $SIF ]]; then
  echo "image exists, skipping pull: $SIF"
else
  singularity pull "$SIF" "$SIF_SOURCE"
fi
tx python -c "import sys, torch; print('container python', sys.version.split()[0], '| torch', torch.__version__, '| cuda build', torch.version.cuda)"

step "2. Pinned Python packages"
tx test -r "$REPO_DIR/env/versions/requirements.lock" \
  || { echo "the repository is not visible in the container; add it to BIND_PATHS"; exit 1; }
tx python -m pip install --user --no-warn-script-location \
  -r "$REPO_DIR/env/versions/requirements.lock"

step "3. Integrity checks"
# torch must come from the image, not from a second copy in $PYUSER
if compgen -G "$PYUSER/lib/python*/site-packages/torch" >/dev/null; then
  echo "ERROR: pip installed a second torch into $PYUSER; remove it and re-run"
  exit 1
fi
tx python -c "import torch; print('torch loaded from', torch.__file__)"
tx python -m pip check

step "4. Model"
if [[ -s $MODEL_DIR/model_checkpoint.pt ]]; then
  echo "model present, skipping download: $MODEL_DIR"
else
  tx python -c "
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ['MODEL_REPO'],
                        revision=os.environ['MODEL_REVISION'],
                        local_dir=os.environ['MODEL_DIR']))"
fi
ls -lh "$MODEL_DIR"
GOT=$(sha256sum "$MODEL_DIR/model_checkpoint.pt" | cut -d' ' -f1)
if [[ $GOT != "$MODEL_SHA256" ]]; then
  echo "ERROR: model_checkpoint.pt sha256 $GOT differs from the pinned $MODEL_SHA256"
  exit 1
fi
echo "model checksum matches the pinned value"

step "5. Record versions"
tx python -m pip freeze > "$SOFTWARE_DIR/logs/pip_freeze_${STAMP}.txt"
singularity inspect "$SIF" > "$SOFTWARE_DIR/logs/image_inspect_${STAMP}.txt" || true
sha256sum "$SIF" > "$SOFTWARE_DIR/logs/image_sha256_${STAMP}.txt"
echo "written to $SOFTWARE_DIR/logs/"

singularity cache clean -f
rm -rf "$SOFTWARE_DIR"/sing/pip/* "$SOFTWARE_DIR"/sing/tmp/*

step "DONE -- next: bash env/check_config.sh"
