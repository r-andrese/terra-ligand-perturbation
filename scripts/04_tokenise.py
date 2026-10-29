#!/usr/bin/env python
"""
Step 04 (Yan Xenium) - TERRA tokenisation, one patient per task
File: scripts/04_tokenise.py

Purpose
  Harmonise and tokenise each included patient ONCE, so that every knockout
  in 06 is built at token level from the same baseline sequences (no
  re-tokenisation, which reshuffles tied genes; see README). Also records,
  for every CSF1+ fibroblast, its own 256-token block, which 05 uses to
  choose rank-matched null genes.

Definitions
  CSF1+ fibroblast   cell_type "Fibro", cell_state in the source states,
                     raw CSF1 count > 0
  Target macrophages cell_type "Mye", cell_state in the target states
  (source and target states are read from config/analysis.cfg)
  Depth batch        median total counts of annotated cells (cell_state not
                     missing), shared panel, before harmonisation:
                     "deep" if >= 500, otherwise "shallow"

Pipeline
  harmonize_adata once per patient (min_genes_per_cell = 10,
  min_cells_per_gene = 0, bundle dictionaries); obs['cell_id'] = obs_names
  ("<patient>_<xenium id>"; 02 keeps the original Xenium id there and it
  repeats across patients); tokenize_adata per tissue piece (obs['sample'])
  with neighbour ids, concatenated. TERRA-112M uses the "cell_graph" layout:
  11 blocks x 256 tokens (the cell, then its 10 nearest neighbours,
  nearest first).

Patients
  All patients exported by 01/02 except those excluded in 03:
  P10, P23 (median depth < 100) and P71, P87, P89, P93 (< 50 CSF1+
  fibroblasts). The script stops unless exactly 38 remain. The task picks
  patient number SGE_TASK_ID from the sorted list; PATIENT=<id> overrides.

Reads : <WORK_DIR>/export/<PATIENT>/adata.h5ad (from 02_build_h5ad.py)
Writes: <KO_DIR>/<PATIENT>/tokens/        tokenised Hugging Face dataset
        <KO_DIR>/<PATIENT>/meta.npz       per-row annotation, neighbour index
        <KO_DIR>/<PATIENT>/cf_blocks.npz  own 256-token block of each CSF1+ fib
        <RESULTS_DIR>/04_tokenise/04_tokenise_<PATIENT>.xlsx  (unformatted)
        (KO_DIR = <WORK_DIR>/terra_ko)

FORCE_REBUILD = False skips patients whose outputs already exist.
Paths come from config/paths.cfg via env/terra_env.sh.
Runs inside the TERRA container (CPU only), via scripts/04_tokenise.sh.
"""

import logging
import os
import shutil
import sys
import time
from pathlib import Path

EXP_DIR = Path(os.environ["WORK_DIR"]) / "export"
RES_DIR = Path(os.environ["RESULTS_DIR"]) / "04_tokenise"
KO_DIR = Path(os.environ["WORK_DIR"]) / "terra_ko"
MODEL = Path(os.environ["MODEL_DIR"])            # set by terra_env.sh
EXCLUDED = {"P10": "median depth < 100", "P23": "median depth < 100",
            "P71": "< 50 CSF1+ fibroblasts", "P87": "< 50 CSF1+ fibroblasts",
            "P89": "< 50 CSF1+ fibroblasts", "P93": "< 50 CSF1+ fibroblasts"}
N_INCLUDED = 38
FORCE_REBUILD = False
NPROC = int(os.environ.get("NSLOTS", "1"))
DEPTH_CUT = 500

COL_TYPE, COL_STATE, COL_SAMPLE = "cell_type", "cell_state", "sample"
FIB_TYPE = "Fibro"
from analysis_config import load_states   # noqa: E402
FIB_STATES, MAC_STATES = load_states()
MAC_TYPE = "Mye"
GENE = "CSF1"
W, L_CELL, N_SEG = 2816, 256, 11      # model_input_size, block length, blocks

RES_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", str(KO_DIR / "hf_cache"))
os.environ.setdefault("HF_HOME", str(KO_DIR / "hf_home"))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import pyarrow as pa                    # noqa: E402
import pyarrow.compute as pc            # noqa: E402
import yaml                             # noqa: E402
import pickle                           # noqa: E402
import anndata as ad                    # noqa: E402
import scipy.sparse as sp               # noqa: E402
from datasets import concatenate_datasets  # noqa: E402

from terra.inference.harmonize import harmonize_adata      # noqa: E402
from terra.inference.tokenize import tokenize_adata        # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("04")
CHECKS, TIMES = [], {}


def check(name, value, expected, passed):
    CHECKS.append({"check": name, "value": value, "expected": expected,
                   "pass": bool(passed)})
    log.info("CHECK %-55s value=%s expected=%s pass=%s",
             name, value, expected, passed)


def included_patients():
    pts = sorted(d.name for d in EXP_DIR.iterdir()
                 if d.is_dir() and (d / "adata.h5ad").exists())
    inc = [p for p in pts if p not in EXCLUDED]
    if len(inc) != N_INCLUDED:
        sys.exit(f"expected {N_INCLUDED} included patients, found {len(inc)}")
    return inc


def pick_patient(pts):
    if os.environ.get("PATIENT"):
        p = os.environ["PATIENT"]
        if p not in pts:
            sys.exit(f"PATIENT={p} is not an included patient")
        return p
    task = os.environ.get("SGE_TASK_ID", "undefined")
    if not task.isdigit():
        sys.exit("set PATIENT or run as an SGE array task (-t 1-38)")
    return pts[int(task) - 1]


def restore_format(ds):
    cols = [c for c in ds.column_names if c not in ("cell_id", "cell_ids")]
    ds.set_format(type="torch", columns=cols, output_all_columns=True)
    return ds


def read_list_column(ds, col, width, dtype, batch_size=20000):
    out = np.empty((len(ds), width), dtype=dtype)
    pos = 0
    for tbl in ds.with_format("arrow").iter(batch_size=batch_size):
        arr = tbl.column(col).combine_chunks()
        n = len(arr)
        flat = pc.list_flatten(arr).to_numpy(zero_copy_only=False)
        if flat.size != n * width:
            raise ValueError(f"{col}: width is not {width}")
        out[pos:pos + n] = flat.reshape(n, width).astype(dtype, copy=False)
        pos += n
    return out


def read_block_ids(ds, width, batch_size=20000):
    # cell_ids repeats one id L_CELL times per block; read first and last
    # position of every block and require them to agree
    starts = np.arange(N_SEG) * L_CELL
    ends = starts + L_CELL - 1
    out = np.empty((len(ds), N_SEG), dtype=object)
    bad, pos = 0, 0
    for tbl in ds.with_format("arrow").iter(batch_size=batch_size):
        arr = tbl.column("cell_ids").combine_chunks()
        n = len(arr)
        flat = pc.list_flatten(arr)
        base = (np.arange(n) * width)[:, None]
        a = pc.take(flat, pa.array((base + starts).ravel())).to_numpy(
            zero_copy_only=False).reshape(n, N_SEG)
        b = pc.take(flat, pa.array((base + ends).ravel())).to_numpy(
            zero_copy_only=False).reshape(n, N_SEG)
        bad += int((a != b).sum())
        out[pos:pos + n] = a
        pos += n
    return out, bad


T0 = time.perf_counter()
pts = included_patients()
PATIENT = pick_patient(pts)
WORK = KO_DIR / PATIENT
WORK.mkdir(parents=True, exist_ok=True)
XLSX = RES_DIR / f"04_tokenise_{PATIENT}.xlsx"
done = [WORK / "tokens" / "dataset_info.json", WORK / "meta.npz",
        WORK / "cf_blocks.npz", XLSX]
if all(f.exists() for f in done) and not FORCE_REBUILD:
    print(f"{PATIENT}: outputs exist, skipped (FORCE_REBUILD = False)", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# 1. Model config and gene token
# ---------------------------------------------------------------------------
print(f"[1] {PATIENT}: model config", flush=True)
with open(MODEL / "model_config.yaml") as fh:
    dcfg = yaml.safe_load(fh)["data"]
for k, v in [("tokenizer_type", "cell_graph"), ("model_input_size", W),
             ("n_neighs", N_SEG - 1), ("seq_len_cell", L_CELL),
             ("seq_len_neighborhood", L_CELL * (N_SEG - 1))]:
    check(f"model config {k}", dcfg.get(k), v, dcfg.get(k) == v)
if not all(c["pass"] for c in CHECKS):
    sys.exit("Model config does not match the 11 x 256 cell_graph layout")
with open(MODEL / "token_dictionary.pkl", "rb") as fh:
    token_dict = pickle.load(fh)

# ---------------------------------------------------------------------------
# 2. Read, depth, harmonise once
# ---------------------------------------------------------------------------
print("[2] read, depth batch, harmonise", flush=True)
t0 = time.perf_counter()
adata = ad.read_h5ad(EXP_DIR / PATIENT / "adata.h5ad")
for c in (COL_TYPE, COL_STATE, COL_SAMPLE):
    if c not in adata.obs:
        sys.exit(f"obs is missing {c}")
n_raw = adata.n_obs
annotated = adata.obs[COL_STATE].notna().to_numpy()
ncount = np.asarray(adata.X.sum(axis=1)).ravel()
depth_median = float(np.median(ncount[annotated]))
depth_batch = "deep" if depth_median >= DEPTH_CUT else "shallow"
adata.obs["cell_id"] = adata.obs_names.astype(str)

adata_h = harmonize_adata(
    adata,
    gene_mapping_dict_file_path=str(MODEL / "ensembl_dictionary.pkl"),
    gene_occurrence_count_file_path=str(MODEL / "gene_count_dictionary.pkl"),
    min_genes_per_cell=10, min_cells_per_gene=0)
del adata
adata_h.obs[COL_SAMPLE] = adata_h.obs[COL_SAMPLE].astype(str)
TIMES["read_harmonise"] = time.perf_counter() - t0

if GENE not in adata_h.var_names:
    sys.exit(f"{GENE} removed by harmonisation")
csf1_col = int(np.where(adata_h.var_names == GENE)[0][0])
csf1_ens = adata_h.var.loc[GENE, "ensembl_id"]
csf1_tok = int(token_dict[csf1_ens])
X = adata_h.X.tocsr() if sp.issparse(adata_h.X) else sp.csr_matrix(adata_h.X)
csf1_raw = np.asarray(X[:, csf1_col].todense()).ravel()
ctype = adata_h.obs[COL_TYPE].astype(str).to_numpy()
cstate = adata_h.obs[COL_STATE].astype(str).to_numpy()
is_cf_obs = (ctype == FIB_TYPE) & np.isin(cstate, FIB_STATES) & (csf1_raw > 0)
is_mac_obs = (ctype == MAC_TYPE) & np.isin(cstate, MAC_STATES)
pieces = sorted(adata_h.obs[COL_SAMPLE].unique())
check("cell_id unique", int(adata_h.obs["cell_id"].duplicated().sum()), 0,
      not adata_h.obs["cell_id"].duplicated().any())
check("cells removed by harmonisation", n_raw - adata_h.n_obs, "report", True)

# ---------------------------------------------------------------------------
# 3. Tokenise per piece and save
# ---------------------------------------------------------------------------
print(f"[3] tokenise {len(pieces)} piece(s)", flush=True)
t0 = time.perf_counter()
cache_root = WORK / "tok_cache"
parts = []
for s in pieces:
    sub = adata_h[adata_h.obs[COL_SAMPLE].to_numpy() == s].copy()
    cache = cache_root / s
    cache.mkdir(parents=True, exist_ok=True)
    ds = tokenize_adata(sub, model_folder_path=str(MODEL),
                        cache_directory_path=str(cache), nproc=NPROC,
                        add_neigh_cell_ids=True, include_special_tokens=False)
    if len(ds) != sub.n_obs:
        sys.exit(f"{s}: tokenised {len(ds)} rows for {sub.n_obs} cells")
    parts.append(ds)
ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
ds = restore_format(ds)
if (WORK / "tokens").exists():
    shutil.rmtree(WORK / "tokens")
ds.save_to_disk(str(WORK / "tokens"))
TIMES["tokenise_save"] = time.perf_counter() - t0
del ds, parts

# Reload from the saved copy (what 06 will read) and drop the build cache
from datasets import load_from_disk     # noqa: E402
ds = restore_format(load_from_disk(str(WORK / "tokens")))
shutil.rmtree(cache_root, ignore_errors=True)

# ---------------------------------------------------------------------------
# 4. Row annotation, neighbour index, CSF1+ fibroblast blocks
# ---------------------------------------------------------------------------
print("[4] neighbour index and CSF1+ fibroblast blocks", flush=True)
t0 = time.perf_counter()
ids = np.asarray([str(x) for x in ds.with_format(None)["cell_id"]])
row2obs = pd.Index(adata_h.obs["cell_id"].astype(str)).get_indexer(ids)
check("every row maps to a harmonised cell", int((row2obs < 0).sum()), 0,
      (row2obs >= 0).all())
n = len(ids)
tok = read_list_column(ds, "gene_tokens", W, np.int32)
blk_ids, misaligned = read_block_ids(ds, W)
check("cell_ids constant within every block", misaligned, 0, misaligned == 0)
nb_idx = pd.Index(ids).get_indexer(blk_ids.ravel()).reshape(n, N_SEG)
check("all block ids resolve to rows", int((nb_idx < 0).sum()), 0,
      (nb_idx >= 0).all())
check("block 0 is the row's own cell", int((nb_idx[:, 0] != np.arange(n)).sum()),
      0, (nb_idx[:, 0] == np.arange(n)).all())
nb_idx = nb_idx.astype(np.int32)

r_cf = is_cf_obs[row2obs]
r_mac = is_mac_obs[row2obs]
r_type = ctype[row2obs]
r_state = cstate[row2obs]
r_piece = adata_h.obs[COL_SAMPLE].to_numpy()[row2obs]
tok3 = tok.reshape(n, N_SEG, L_CELL)
cf_blocks = tok3[r_cf, 0, :].copy()
own_csf1 = (cf_blocks == csf1_tok).any(axis=1)
nb_cf = r_cf[nb_idx[:, 1:]]
blk_csf1 = (tok3[:, 1:, :] == csf1_tok).any(axis=2)
mac_cf = nb_cf[r_mac].any(axis=1)
mac_csf1 = (nb_cf & blk_csf1)[r_mac].any(axis=1)

var_symbol = adata_h.var_names.astype(str).to_numpy()
var_ens = adata_h.var["ensembl_id"].astype(str).to_numpy()
var_tok = np.array([int(token_dict.get(e, -1)) for e in var_ens], dtype=np.int64)

np.savez(WORK / "meta.npz",
         cell_id=ids.astype(object), piece=r_piece.astype(object),
         cell_type=r_type.astype(object), cell_state=r_state.astype(object),
         is_cf=r_cf, is_mac=r_mac, is_mye=(r_type == MAC_TYPE),
         nb_idx=nb_idx, csf1_raw=csf1_raw[row2obs].astype(np.float32),
         csf1_token=np.int64(csf1_tok), var_symbol=var_symbol.astype(object),
         var_ensembl=var_ens.astype(object), var_token=var_tok,
         depth_median=np.float64(depth_median),
         depth_batch=np.array(depth_batch, dtype=object))
np.savez(WORK / "cf_blocks.npz", cf_tokens=cf_blocks.astype(np.int32),
         cf_cell_id=ids[r_cf].astype(object))
TIMES["index_blocks"] = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 5. Summary workbook
# ---------------------------------------------------------------------------
print("[5] write summary", flush=True)
TIMES["total"] = time.perf_counter() - T0
summary = pd.DataFrame([{
    "patient": PATIENT, "pieces": ";".join(pieces), "n_cells": n,
    "depth_median_annotated": depth_median, "depth_batch": depth_batch,
    "n_csf1pos_fib": int(r_cf.sum()),
    "n_csf1pos_fib_csf1_in_block": int(own_csf1.sum()),
    "frac_csf1pos_fib_csf1_in_block": float(own_csf1.mean()) if own_csf1.size else np.nan,
    "n_target_mac": int(r_mac.sum()),
    **{"n_target_mac_state_%d" % (k + 1): int((r_mac & (r_state == s)).sum())
       for k, s in enumerate(MAC_STATES)},
    "n_target_mac_with_csf1pos_fib_neighbour": int(mac_cf.sum()),
    "n_target_mac_with_csf1_in_fib_neighbour_block": int(mac_csf1.sum()),
    "n_mye": int((r_type == MAC_TYPE).sum())}])
timing = pd.DataFrame({"stage": list(TIMES), "seconds": list(TIMES.values())})

import pandas.io.formats.excel as pdx   # noqa: E402
pdx.ExcelFormatter.header_style = None
with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
    summary.to_excel(xw, sheet_name="summary", index=False)
    pd.DataFrame(CHECKS).to_excel(xw, sheet_name="checks", index=False)
    timing.to_excel(xw, sheet_name="timing", index=False)
failed = [c["check"] for c in CHECKS if not c["pass"]]
print(f"[done] {PATIENT}: {n} cells, {depth_batch}, checks failed: "
      f"{failed if failed else 'none'} ({TIMES['total']:.0f} s)", flush=True)
if failed:
    sys.exit(1)
