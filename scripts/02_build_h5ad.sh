#!/bin/bash
#$ -N tlp_02_build_h5ad
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l tmem=64G,h_vmem=64G
#$ -l h_rt=06:00:00
# -----------------------------------------------------------------------------
# 02_build_h5ad.sh - CPU, single job, after 01_export.sh:
#   qsub -o logs/ -hold_jid tlp_01_export scripts/02_build_h5ad.sh
# Adapt resource names (tmem, h_vmem, gpu) to your scheduler.
# -----------------------------------------------------------------------------
set -euo pipefail
source env/terra_env.sh
require_avx
echo "host: $(hostname)  job: ${JOB_ID:-NA}  start: $(date)"
tx python "$REPO_DIR/scripts/02_build_h5ad.py"
echo "end: $(date)"
