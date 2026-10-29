#!/bin/bash
#$ -N tlp_03_sample_table
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=02:00:00
# -----------------------------------------------------------------------------
# 03_sample_table.sh - CPU, single job, after 02_build_h5ad.sh:
#   qsub -o logs/ -hold_jid tlp_02_build_h5ad scripts/03_sample_table.sh
# Adapt resource names (tmem, h_vmem, gpu) to your scheduler.
# -----------------------------------------------------------------------------
set -euo pipefail
source env/terra_env.sh
require_avx
echo "host: $(hostname)  job: ${JOB_ID:-NA}  start: $(date)"
tx python "$REPO_DIR/scripts/03_sample_table.py"
echo "end: $(date)"
