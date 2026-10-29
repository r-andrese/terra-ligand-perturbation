#!/bin/bash
#$ -N tlp_06_ko_embed
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l tmem=32G,h_vmem=200G
#$ -l h_rt=03:00:00
#$ -l gpu=true
#$ -t 1-38
#$ -tc 8
# -----------------------------------------------------------------------------
# 06_ko_embed.sh - GPU array, one included patient per task (38). Needs a card
# with >= 40 GB: the batch size of 256 needs about 21 GB plus CUDA overhead and 
# fails on 24 GB cards; it is kept fixed so that every patient is embedded identically. 
# Submit from the repository root, after 05_null_genes.sh:
#   qsub -o logs/ -t 1-1 -v PATIENT=P12 scripts/06_ko_embed.sh   one patient first
#   qsub -o logs/ scripts/06_ko_embed.sh                         all (finished
#                                                       patients are skipped)
# Adapt resource names (tmem, h_vmem, gpu) to your scheduler.
# -----------------------------------------------------------------------------
set -euo pipefail
source env/terra_env.sh
require_avx
echo "host: $(hostname)  job: ${JOB_ID:-NA}  task: ${SGE_TASK_ID:-NA}  patient override: ${PATIENT:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
txg python "$REPO_DIR/scripts/06_ko_embed.py"
