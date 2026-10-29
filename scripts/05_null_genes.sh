#!/bin/bash
#$ -N tlp_05_null_genes
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l tmem=16G,h_vmem=16G
#$ -l h_rt=01:00:00
# -----------------------------------------------------------------------------
# 05_null_genes.sh - CPU, single job, after every 04 task has finished:
#   qsub -o logs/ -hold_jid tlp_04_tokenise scripts/05_null_genes.sh
# Adapt resource names (tmem, h_vmem, gpu) to your scheduler.
# -----------------------------------------------------------------------------
set -euo pipefail
source env/terra_env.sh
require_avx
tx python "$REPO_DIR/scripts/05_null_genes.py"
