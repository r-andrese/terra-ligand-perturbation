#!/bin/bash
# =============================================================================
# check_config.sh -- verify configuration, software, input and model
#
# Run from the repository root after env/install.sh:
#     bash env/check_config.sh
# On a login node it checks the configuration, software, input folders, the
# Python environment and the model. Run the same command inside an
# interactive GPU session (e.g. qrsh -l gpu=true ...) to add the GPU checks.
# Prints one line per check and exits 1 if any check fails. It creates WORK_DIR 
# and RESULTS_DIR when they do not exist, and sourcing env/terra_env.sh 
# creates its cache folders under SOFTWARE_DIR and WORK_DIR.
# =============================================================================
set -uo pipefail

source env/terra_env.sh || exit 1

N_FAIL=0
ok()   { printf "  OK    %s\n" "$*"; }
bad()  { printf "  FAIL  %s\n" "$*"; N_FAIL=$((N_FAIL + 1)); }
warn() { printf "  WARN  %s\n" "$*"; }

# -----------------------------------------------------------------------------
echo "[1] Configuration ($TERRA_CFG)"
for k in INPUT_DIR SOFTWARE_DIR WORK_DIR RESULTS_DIR; do
  v=${!k}
  if [[ $v != /* ]]; then bad "$k is not an absolute path: $v"; continue; fi
  [[ $v == *,* ]] && bad "$k contains a comma, which the container environment cannot carry"
  [[ $k == WORK_DIR || $k == RESULTS_DIR ]] && mkdir -p "$v" 2>/dev/null
  if [[ -d $v ]]; then ok "$k = $v"; else bad "$k does not exist: $v"; fi
done

IFS=',' read -ra BINDS <<< "$BIND_PATHS"
for b in "${BINDS[@]}"; do
  [[ $b == *:* ]] && warn "bind entry $b maps to a different container path; the checks below assume host path = container path"
done
is_bound() {
  local p=$1 b src
  for b in "${BINDS[@]}"; do
    src=${b%%:*}; src=${src%/}
    [[ $p == "$src" || $p == "$src"/* ]] && return 0
  done
  return 1
}
for k in REPO_DIR INPUT_DIR SOFTWARE_DIR WORK_DIR RESULTS_DIR; do
  if is_bound "${!k}"; then ok "$k is inside BIND_PATHS"
  else bad "$k (${!k}) is not inside any BIND_PATHS entry"; fi
done

ACFG=$REPO_DIR/config/analysis.cfg
if [[ ! -f $ACFG ]]; then
  bad "config/analysis.cfg missing (copy config/analysis.cfg.example and set the cell states)"
elif grep -qE '^(SOURCE|TARGET)_STATES=.*<' "$ACFG"; then
  bad "config/analysis.cfg still holds placeholders"
elif ! grep -qE '^SOURCE_STATES=.+' "$ACFG" || ! grep -qE '^TARGET_STATES=[^,]+,[^,]+$' "$ACFG"; then
  bad "config/analysis.cfg: set SOURCE_STATES and exactly two TARGET_STATES"
else
  ok "config/analysis.cfg sets the source and target states"
fi

# -----------------------------------------------------------------------------
echo "[2] Software on this node"
if command -v singularity >/dev/null; then ok "$(singularity --version 2>&1)"
else bad "singularity not found on PATH"; fi
if [[ -s $SIF ]]; then ok "container image $SIF"
else bad "container image missing: $SIF (run env/install.sh)"; fi
if command -v Rscript >/dev/null; then
  if Rscript -e 'suppressPackageStartupMessages({library(Matrix); library(igraph); library(openxlsx)})' >/dev/null 2>&1; then
    ok "R with Matrix, igraph and openxlsx (export step)"
  else
    bad "R found, but Matrix, igraph or openxlsx is missing (needed by the export step)"
  fi
else
  warn "Rscript not on PATH here; the export step needs R with Matrix, igraph and openxlsx"
fi
if grep -q -w avx /proc/cpuinfo; then ok "CPU supports AVX"
else warn "this node lacks AVX; container jobs cannot run here"; fi

# -----------------------------------------------------------------------------
echo "[3] Input folders ($INPUT_DIR)"
n_pat=0
for d in "$INPUT_DIR"/*/; do
  [[ -d $d ]] || continue
  n_pat=$((n_pat + 1))
  for f in mat_counts.rds df_meta.rds spatial_xy.rds; do
    [[ -s $d$f ]] || bad "missing $d$f"
  done
done
if (( n_pat == 0 )); then bad "no patient folders in INPUT_DIR"
elif (( n_pat != 44 )); then warn "$n_pat patient folders (the original analysis used 44)"
else ok "44 patient folders"; fi

# -----------------------------------------------------------------------------
if [[ -s $SIF ]]; then
  tx python - <<'PY' || N_FAIL=$((N_FAIL + 1))
import hashlib
import importlib
import importlib.metadata as md
import os
import sys
from pathlib import Path

fails = 0


def ok(m):
    print("  OK    " + m)


def bad(m):
    global fails
    fails += 1
    print("  FAIL  " + m)


print("[4] Python environment in the container (python %s)" % sys.version.split()[0])
lock = Path(os.environ["REPO_DIR"]) / "env/versions/requirements.lock"
want = {}
for line in lock.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        name, ver = line.split("==")
        want[name] = ver
diff = []
for name, ver in sorted(want.items()):
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = "missing"
    if got != ver:
        diff.append("%s %s (lock %s)" % (name, got, ver))
if diff:
    bad("%d package(s) differ from requirements.lock: %s%s"
        % (len(diff), "; ".join(diff[:10]), " ..." if len(diff) > 10 else ""))
else:
    ok("all %d packages match requirements.lock" % len(want))
try:
    import torch
    if torch.__version__ == "2.6.0+cu124":
        ok("torch %s from the image" % torch.__version__)
    else:
        bad("torch %s, expected 2.6.0+cu124 from the image" % torch.__version__)
except ImportError as e:
    bad("torch does not import: %s" % e)
for mod in ("terra", "geomloss", "openpyxl", "sklearn", "datasets", "anndata"):
    try:
        importlib.import_module(mod)
        ok("import %s" % mod)
    except Exception as e:  # noqa: BLE001 (report any import failure)
        bad("import %s: %s" % (mod, e))

print("[5] Model (%s)" % os.environ["MODEL_DIR"])
mdir = Path(os.environ["MODEL_DIR"])
need = ["model_checkpoint.pt", "model_config.yaml", "token_dictionary.pkl",
        "ensembl_dictionary.pkl", "gene_count_dictionary.pkl"]
missing = [f for f in need if not (mdir / f).exists()]
if missing:
    bad("missing model files: %s" % ", ".join(missing))
else:
    ok("model files present")
meta = mdir / ".cache/huggingface/download/model_checkpoint.pt.metadata"
rev = meta.read_text().splitlines()[0].strip() if meta.exists() else "unknown"
if rev == os.environ["MODEL_REVISION"]:
    ok("revision %s" % rev)
else:
    bad("revision %s, pinned %s" % (rev, os.environ["MODEL_REVISION"]))
ckpt = mdir / "model_checkpoint.pt"
if ckpt.exists():
    h = hashlib.sha256()
    with open(ckpt, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            h.update(chunk)
    if h.hexdigest() == os.environ["MODEL_SHA256"]:
        ok("model_checkpoint.pt sha256 matches the pinned value")
    else:
        bad("model_checkpoint.pt sha256 %s differs from the pinned value" % h.hexdigest())
sys.exit(1 if fails else 0)
PY
else
  warn "container image missing; Python and model checks skipped"
fi

# -----------------------------------------------------------------------------
if [[ -n ${CUDA_VISIBLE_DEVICES:-} && -s $SIF ]]; then
  echo "[6] GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
  txg python - <<'PY' || N_FAIL=$((N_FAIL + 1))
import sys

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    print("  FAIL  CUDA not visible inside the container")
    sys.exit(1)
fails = 0
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
print("  OK    %s, %.0f GiB" % (name, mem))
# The knockout step embeds with a fixed batch size of 256 (about 21 GB plus
# CUDA overhead); cards below 40 GB fail.
if mem < 39:
    print("  FAIL  the knockout step needs a card with >= 40 GB")
    fails += 1
if torch.cuda.is_bf16_supported():
    print("  OK    bf16 supported")
else:
    print("  FAIL  bf16 not supported on this card")
    fails += 1
# The attention call TERRA inference uses, at its sequence length (2816)
q = torch.randn(4, 8, 2816, 64, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = F.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize()
print("  OK    bf16 attention at sequence length 2816: %s" % (tuple(out.shape),))
sys.exit(1 if fails else 0)
PY
else
  warn "not in a GPU session; GPU checks skipped"
fi

# -----------------------------------------------------------------------------
echo
if (( N_FAIL == 0 )); then
  echo "ALL CHECKS PASSED"
else
  echo "$N_FAIL check(s) failed"
  exit 1
fi
