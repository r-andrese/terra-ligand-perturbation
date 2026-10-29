#!/bin/bash
#$ -N tlp_01_export
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l tmem=64G,h_vmem=64G
#$ -l h_rt=06:00:00
# -----------------------------------------------------------------------------
# 01_export.sh - CPU, single job. Runs R on the host. 
# R needs the packages Matrix, igraph and openxlsx. 
# Submit from the repository root:
#   qsub -o logs/ scripts/01_export.sh
# Adapt resource names (tmem, h_vmem, gpu) to your scheduler.
# -----------------------------------------------------------------------------
set -euo pipefail
source env/terra_env.sh
echo "host: $(hostname)  job: ${JOB_ID:-NA}  start: $(date)"
Rscript "$REPO_DIR/scripts/01_export.R"
echo "end: $(date)"
