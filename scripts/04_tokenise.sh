#!/bin/bash
#$ -N tlp_04_tokenise
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l tmem=64G,h_vmem=64G
#$ -l h_rt=06:00:00
#$ -t 1-38
# -----------------------------------------------------------------------------
# 04_tokenise.sh - CPU array, one included patient per task.
# Submit from the repository root:
#   qsub -o logs/ scripts/04_tokenise.sh                         all patients
#   qsub -o logs/ -t 1-1 -v PATIENT=P12 scripts/04_tokenise.sh   one patient
# Adapt resource names (tmem, h_vmem, gpu) to your scheduler.
# -----------------------------------------------------------------------------
set -euo pipefail
source env/terra_env.sh
require_avx
echo "host: $(hostname)  job: ${JOB_ID:-NA}  task: ${SGE_TASK_ID:-NA}  patient override: ${PATIENT:-none}"
tx python "$REPO_DIR/scripts/04_tokenise.py"
