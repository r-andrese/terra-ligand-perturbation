#!/bin/bash
#$ -N tlp_07_ko_pool
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=24:00:00
# -----------------------------------------------------------------------------
# 07_ko_pool.sh - CPU, single job, after every 06 task has finished:
#   qsub -o logs/ -hold_jid tlp_06_ko_embed scripts/07_ko_pool.sh
# Pools the per-patient knockout files, fits the leave-one-donor-out
# classifiers and writes the unformatted workbook. 
# Adapt resource names (tmem, h_vmem, gpu) to your scheduler.
# -----------------------------------------------------------------------------
set -euo pipefail
source env/terra_env.sh
require_avx
echo "host: $(hostname)  job: ${JOB_ID:-NA}  start: $(date)"
tx python "$REPO_DIR/scripts/07_ko_pool.py"
echo "end: $(date)"
