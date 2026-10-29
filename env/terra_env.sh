# =============================================================================
# terra_env.sh -- shared settings for running TERRA inside Singularity
#
# Sourced by every job script and interactive session, from the
# repository root:
#     source env/terra_env.sh
# It reads config/paths.cfg, defines the derived locations and the pinned
# model, and provides:
#     tx  CMD        run CMD in the container (CPU)
#     txg CMD        run CMD in the container with GPU access, Hugging Face
#                    offline
#     require_avx    stop a job early on nodes whose CPU lacks AVX
# All paths reach Python through the container environment (TERRA_CENV), so
# the scripts contain no paths of their own. 
# =============================================================================

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TERRA_CFG=${TERRA_CFG:-$REPO_DIR/config/paths.cfg}
if [[ ! -f $TERRA_CFG ]]; then
  echo "terra_env.sh: $TERRA_CFG not found; copy config/paths.cfg.example" \
       "to config/paths.cfg and edit it" >&2
  return 1
fi
set -a
source "$TERRA_CFG"
set +a
for _k in INPUT_DIR WORK_DIR RESULTS_DIR SOFTWARE_DIR BIND_PATHS; do
  if [[ -z ${!_k:-} ]]; then
    echo "terra_env.sh: $_k is not set in $TERRA_CFG" >&2
    return 1
  fi
done
unset _k
export REPO_DIR

# Container image (pulled once by env/install.sh) and the Python packages
# installed into it with pip --user
export SIF_SOURCE=docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
export SIF=$SOFTWARE_DIR/containers/pytorch_2.6.0_cu124.sif
export PYUSER=$SOFTWARE_DIR/pyuser

# Pretrained model, pinned to the revision used for the original analysis
export MODEL_REPO=Lotfollahi-lab/TERRA-112M
export MODEL_REVISION=00d1c977e49364ba10c28692abcbd102e115bf91
export MODEL_SHA256=d6b46ad4b8e63e67b968e789256839da80dafac3e5de482a5e21a5e5eb4bada2
export MODEL_DIR=$SOFTWARE_DIR/models/TERRA-112M

# Singularity build cache and temporary space (kept off the home directory)
export SINGULARITY_CACHEDIR=$SOFTWARE_DIR/sing/cache
export SINGULARITY_TMPDIR=$SOFTWARE_DIR/sing/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$SOFTWARE_DIR/sing/pip" \
         "$SOFTWARE_DIR"/cache/{hf,mpl,numba} "$WORK_DIR"/hf_datasets

# Environment passed INTO the container. --cleanenv drops the host
# environment, every cache that would default to ~ is redirected.
TERRA_CENV="PYTHONUSERBASE=$PYUSER"
TERRA_CENV+=",PIP_CACHE_DIR=$SOFTWARE_DIR/sing/pip"
TERRA_CENV+=",HF_HOME=$SOFTWARE_DIR/cache/hf"
TERRA_CENV+=",MPLCONFIGDIR=$SOFTWARE_DIR/cache/mpl"
TERRA_CENV+=",NUMBA_CACHE_DIR=$SOFTWARE_DIR/cache/numba"
# Temporary files go to the node-local /tmp
TERRA_CENV+=",TMPDIR=/tmp"
TERRA_CENV+=",HF_DATASETS_CACHE=$WORK_DIR/hf_datasets"
for _k in REPO_DIR INPUT_DIR WORK_DIR RESULTS_DIR \
          MODEL_DIR MODEL_REPO MODEL_REVISION MODEL_SHA256; do
  TERRA_CENV+=",$_k=${!_k}"
done
unset _k
export TERRA_CENV

# Scheduler variables the scripts read (slots, array task, patient override).
# SINGULARITYENV_* variables pass through --cleanenv.
_terra_job_env() {
  export SINGULARITYENV_NSLOTS=${NSLOTS:-1}
  export SINGULARITYENV_OMP_NUM_THREADS=${NSLOTS:-1}
  export SINGULARITYENV_SGE_TASK_ID=${SGE_TASK_ID:-undefined}
  export SINGULARITYENV_PATIENT=${PATIENT:-}
}

tx() {
  _terra_job_env
  singularity exec --cleanenv -B "$BIND_PATHS" --env "$TERRA_CENV" "$SIF" "$@"
}

# GPU allocation: SGE assigns a card by setting CUDA_VISIBLE_DEVICES on the
# host. --cleanenv would drop it, and the container would then see every card
# on the node and use card 0, which may belong to another job. Passing it as
# SINGULARITYENV_CUDA_VISIBLE_DEVICES keeps the assignment, including
# multi-card lists such as "4,5" that the comma-separated --env cannot carry.
txg() {
  if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    echo "txg: CUDA_VISIBLE_DEVICES is not set -- run inside a GPU job" >&2
    return 1
  fi
  _terra_job_env
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  singularity exec --nv --cleanenv -B "$BIND_PATHS" \
    --env "$TERRA_CENV,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1" "$SIF" "$@"
}

# The container's jaxlib (imported by TERRA) needs AVX; some older nodes lack
# it. Fail at once with a clear message instead of an import traceback.
require_avx() {
  if ! grep -q -w avx /proc/cpuinfo; then
    echo "ERROR: $(hostname) has no AVX; resubmit excluding this node" >&2
    exit 1
  fi
}

export -f _terra_job_env tx txg require_avx 2>/dev/null || true
