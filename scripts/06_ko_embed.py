#!/usr/bin/env python
"""
Step 06 (Yan Xenium) - TERRA in silico knockouts and embedding, one
patient per task
File: scripts/06_ko_embed.py

Purpose
  Build every knockout at token level from the baseline sequences saved by
  04 (no re-tokenisation) and embed the target macrophages with TERRA-112M.
  Pooling, the classifier readout and the decision rule are in 07.

Edits (only in the neighbour blocks 1..10 that belong to a CSF1+ fibroblast;
the macrophage's own block and all other neighbours are never touched)
  CSF1_remove   main knockout: the CSF1 token and its value are removed and
                the rest of the block moves up one position, padding at the
                end (TERRA's knockout, value zeroed and token removed,
                restricted to the right blocks)
  CSF1_zero     secondary: CSF1 value set to 0, token kept
  NULL_<gene>   the 20 rank-matched null genes from 05, each removed exactly
                like CSF1_remove, in the same blocks
  POS_block     positive control: the whole CSF1+ fibroblast block is emptied
                (all tokens and values set to 0), as if that neighbour had no
                detected genes
  A cell is edited by a condition when at least one of its CSF1+ fibroblast
  neighbour blocks contains the gene (POS_block: has a CSF1+ fibroblast
  neighbour); n_edit counts those blocks.

Embedding
  baseline: every Mye cell of the patient (classifier training set for 07)
  baseline and every condition: the target macrophages (the target
  states), always the same rows in the same order, so batching is
  identical across conditions. Readouts: spatial_cell_emb and
  neighborhood_emb; cell_emb kept as a negative control (a neighbourhood edit
  cannot change it).

Checks
  each condition changes exactly the cells with n_edit > 0 and nothing
  outside the edited blocks; edited blocks keep their padding at the end;
  cell_emb unchanged; embeddings have one row per sequence.

Reads : <KO_DIR>/<P>/tokens/, <KO_DIR>/<P>/meta.npz (04),
        <RESULTS_DIR>/05_null_genes/05_null_genes.tsv
Writes: <KO_DIR>/<P>/emb_ko.npz          embeddings for 07
        <RES_DIR>/06_ko_embed_<P>.xlsx     (unformatted; per-cell shifts)
        (KO_DIR = <WORK_DIR>/terra_ko; RES_DIR = <RESULTS_DIR>/06_ko_embed)

The task picks patient number SGE_TASK_ID from the sorted list of the 38
included patients; PATIENT=<id> overrides. FORCE_REBUILD = False skips
patients whose outputs already exist.
Paths come from config/paths.cfg via env/terra_env.sh.
Runs inside the TERRA container on a GPU node, via scripts/06_ko_embed.sh.
"""

import logging
import os
import sys
import time
from pathlib import Path

EXP_DIR = Path(os.environ["WORK_DIR"]) / "export"
NULL_TSV = Path(os.environ["RESULTS_DIR"]) / "05_null_genes/05_null_genes.tsv"
RES_DIR = Path(os.environ["RESULTS_DIR"]) / "06_ko_embed"
KO_DIR = Path(os.environ["WORK_DIR"]) / "terra_ko"
MODEL = Path(os.environ["MODEL_DIR"])            # set by terra_env.sh
EXCLUDED = {"P10", "P23", "P71", "P87", "P89", "P93"}
N_INCLUDED = 38
FORCE_REBUILD = False
BATCH_SIZE = 256
NUM_WORKERS = int(os.environ.get("NSLOTS", "1"))
W, L_CELL, N_SEG = 2816, 256, 11
EMB_TYPES = ["spatial_cell_emb", "neighborhood_emb", "cell_emb"]

RES_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", str(KO_DIR / "hf_cache"))
os.environ.setdefault("HF_HOME", str(KO_DIR / "hf_home"))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import pyarrow.compute as pc            # noqa: E402
import torch                            # noqa: E402
from datasets import load_from_disk     # noqa: E402

from terra.inference.embed import embed_dataset            # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("06")
CHECKS, TIMES, GPU_MEM = [], {}, {}


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


def subset(ds, rows):
    # Strip the format before select/flatten: flattening a formatted dataset
    # re-encodes values through the formatter (precision loss)
    out = (ds.with_format(None).select(rows).remove_columns("cell_ids")
           .flatten_indices(keep_in_memory=True))
    return restore_format(out)


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


def apply_edit(tok, expr, blk, token, mode):
    """Edit neighbour blocks 1..10 of every row where blk[row, block-1].

    token None = whole block; mode "remove" zeroes token and value and moves
    the remaining tokens of the block up (stable, so order is kept and padding
    stays at the end); mode "zero" zeroes the value and keeps the token.
    Returns edited copies and n_edit (blocks actually edited per row).
    """
    n = tok.shape[0]
    t, e = tok.copy(), expr.copy()
    nt = t.reshape(n, N_SEG, L_CELL)[:, 1:, :]          # views
    ne = e.reshape(n, N_SEG, L_CELL)[:, 1:, :]
    if token is None:
        sel = blk[:, :, None] & (nt != 0)
    else:
        sel = blk[:, :, None] & (nt == token)
    edited_blk = sel.any(axis=2)                         # (n, 10)
    if mode == "zero":
        ne[sel] = 0.0
    elif mode == "remove":
        nt[sel] = 0
        ne[sel] = 0.0
        r, b = np.nonzero(edited_blk)
        if r.size:
            sub_t, sub_e = nt[r, b, :], ne[r, b, :]
            order = np.argsort(sub_t == 0, axis=1, kind="stable")
            nt[r, b, :] = np.take_along_axis(sub_t, order, axis=1)
            ne[r, b, :] = np.take_along_axis(sub_e, order, axis=1)
    else:
        raise ValueError(mode)
    return t, e, edited_blk.sum(axis=1).astype(np.int16), edited_blk


def replace_arrays(ds, tok, expr):
    body = ds.with_format(None)

    def fn(batch, indices):
        batch["gene_tokens"] = tok[indices]
        batch["gene_expr"] = expr[indices]
        return batch

    out = body.map(fn, batched=True, batch_size=1000, with_indices=True,
                   features=body.features, keep_in_memory=True,
                   load_from_cache_file=False)
    return restore_format(out)


def embed(ds, name):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = embed_dataset(ds, model_folder_path=str(MODEL), batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, include_spatial_cell_emb=True)
    TIMES[f"embed_{name}"] = time.perf_counter() - t0
    if torch.cuda.is_available():
        GPU_MEM[name] = torch.cuda.max_memory_allocated() / 1024 ** 3
    out = {k: np.asarray(out[k], dtype=np.float32) for k in EMB_TYPES}
    for k in EMB_TYPES:
        if out[k].shape[0] != len(ds):
            raise ValueError(f"{name}/{k}: {out[k].shape[0]} rows for {len(ds)}")
    return out


def cos_shift(x, y):
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    yn = y / np.linalg.norm(y, axis=1, keepdims=True)
    return 1.0 - np.sum(xn * yn, axis=1)


T0 = time.perf_counter()
pts = included_patients()
PATIENT = pick_patient(pts)
WORK = KO_DIR / PATIENT
NPZ = WORK / "emb_ko.npz"
XLSX = RES_DIR / f"06_ko_embed_{PATIENT}.xlsx"
if NPZ.exists() and XLSX.exists() and not FORCE_REBUILD:
    print(f"{PATIENT}: outputs exist, skipped (FORCE_REBUILD = False)", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# 1. Inputs
# ---------------------------------------------------------------------------
print(f"[1] {PATIENT}: load tokens, annotation, null genes", flush=True)
meta = np.load(WORK / "meta.npz", allow_pickle=True)
ds = restore_format(load_from_disk(str(WORK / "tokens")))
ids = meta["cell_id"].astype(str)
check("tokens and meta have the same rows", len(ds), len(ids), len(ds) == len(ids))
ds_ids = np.asarray([str(x) for x in ds.with_format(None)["cell_id"]])
check("row order identical", bool((ds_ids == ids).all()), True, (ds_ids == ids).all())
is_cf, is_mac, is_mye = meta["is_cf"], meta["is_mac"], meta["is_mye"]
state, piece = meta["cell_state"].astype(str), meta["piece"].astype(str)
nb_idx = meta["nb_idx"]
csf1_tok = int(meta["csf1_token"])
null = pd.read_csv(NULL_TSV, sep="\t")
check("null genes", len(null), 20, len(null) == 20)

mye_rows = np.where(is_mye)[0]
t_rows = np.where(is_mac)[0]
t_in_mye = np.searchsorted(mye_rows, t_rows)
check("targets are Mye cells", bool((mye_rows[t_in_mye] == t_rows).all()), True,
      (mye_rows[t_in_mye] == t_rows).all())
blk = is_cf[nb_idx[t_rows, 1:]]                     # (n_t, 10) CSF1+ fib blocks
n_cf_nb = blk.sum(axis=1).astype(np.int16)

conds = [("CSF1_remove", csf1_tok, "remove"),
         ("CSF1_zero", csf1_tok, "zero"),
         ("POS_block", None, "remove")]
conds += [(f"NULL_{s}", int(tk), "remove")
          for s, tk in zip(null["symbol"], null["token"])]
cond_names = [c[0] for c in conds]

# ---------------------------------------------------------------------------
# 2. Baseline embeddings
# ---------------------------------------------------------------------------
print(f"[2] baseline: {len(mye_rows)} Mye cells, {len(t_rows)} targets", flush=True)
base_mye = embed(subset(ds, mye_rows), "base_mye")
base_t_ds = subset(ds, t_rows)
tok_t = read_list_column(base_t_ds, "gene_tokens", W, np.int64)
expr_t = read_list_column(base_t_ds, "gene_expr", W, np.float64)
base_t = embed(base_t_ds, "base_targets")
check("own block never contains padding before genes",
      int(((tok_t[:, :L_CELL] == 0)[:, :-1] & (tok_t[:, 1:L_CELL] != 0)).sum()), 0,
      not ((tok_t[:, :L_CELL] == 0)[:, :-1] & (tok_t[:, 1:L_CELL] != 0)).any())

# ---------------------------------------------------------------------------
# 3. Knockouts
# ---------------------------------------------------------------------------
print(f"[3] {len(conds)} knockouts", flush=True)
n_t = len(t_rows)
n_edit = np.zeros((n_t, len(conds)), dtype=np.int16)
emb_sp = np.zeros((len(conds), n_t, base_t["spatial_cell_emb"].shape[1]), np.float32)
emb_nb = np.zeros((len(conds), n_t, base_t["neighborhood_emb"].shape[1]), np.float32)
cell_dev = []
for j, (name, token, mode) in enumerate(conds):
    t2, e2, ne_j, eb = apply_edit(tok_t, expr_t, blk, token, mode)
    n_edit[:, j] = ne_j
    changed = (t2 != tok_t).any(axis=1) | (e2 != expr_t).any(axis=1)
    edited_rows = ne_j > 0
    # positions outside the edited blocks must be untouched
    mask = np.zeros((n_t, N_SEG), bool)
    mask[:, 1:] = eb
    outside = ~np.repeat(mask, L_CELL, axis=1)
    untouched = ((t2 == tok_t) | ~outside).all() and ((e2 == expr_t) | ~outside).all()
    pad_ok = not ((t2[:, L_CELL:].reshape(n_t, N_SEG - 1, L_CELL) == 0)[:, :, :-1]
                  & (t2[:, L_CELL:].reshape(n_t, N_SEG - 1, L_CELL) != 0)[:, :, 1:]).any()
    check(f"{name}: changed cells == edited cells",
          int((changed != edited_rows).sum()), 0, (changed == edited_rows).all())
    check(f"{name}: nothing changed outside edited blocks", bool(untouched), True, untouched)
    check(f"{name}: padding stays at block ends", bool(pad_ok), True, pad_ok)
    out = embed(replace_arrays(base_t_ds, t2, e2), name)
    emb_sp[j] = out["spatial_cell_emb"]
    emb_nb[j] = out["neighborhood_emb"]
    cell_dev.append(float(np.abs(out["cell_emb"] - base_t["cell_emb"]).max()))
    check(f"{name}: cells edited", int(edited_rows.sum()), "report", True)

# ---------------------------------------------------------------------------
# 4. Per-cell shifts and outputs
# ---------------------------------------------------------------------------
print("[4] per-cell shifts and outputs", flush=True)
per_cell = pd.DataFrame({"cell_id": ids[t_rows], "patient": PATIENT,
                         "sample": piece[t_rows], "cell_state": state[t_rows],
                         "n_csf1pos_fib_neighbours": n_cf_nb})
cols = {}
for j, name in enumerate(cond_names):
    cols[f"n_edit_{name}"] = n_edit[:, j]
    cols[f"shift_spatial_{name}"] = cos_shift(emb_sp[j], base_t["spatial_cell_emb"])
    cols[f"shift_neigh_{name}"] = cos_shift(emb_nb[j], base_t["neighborhood_emb"])
per_cell = pd.concat([per_cell, pd.DataFrame(cols)], axis=1)
neg = pd.DataFrame({"condition": cond_names, "max_abs_diff_cell_emb": cell_dev})
for name, v in zip(cond_names, cell_dev):
    check(f"{name}: cell_emb unchanged (max abs diff)", v, "< 1e-4", v < 1e-4)

np.savez(NPZ, patient=np.array(PATIENT, dtype=object),
         depth_batch=meta["depth_batch"],
         mye_cell_id=ids[mye_rows].astype(object),
         mye_state=state[mye_rows].astype(object),
         base_mye_spatial=base_mye["spatial_cell_emb"],
         base_mye_neigh=base_mye["neighborhood_emb"],
         base_mye_cell=base_mye["cell_emb"],
         t_in_mye=t_in_mye, t_cell_id=ids[t_rows].astype(object),
         t_state=state[t_rows].astype(object), n_cf_nb=n_cf_nb,
         base_t_spatial=base_t["spatial_cell_emb"],
         base_t_neigh=base_t["neighborhood_emb"],
         cond_names=np.array(cond_names, dtype=object), n_edit=n_edit,
         cond_spatial=emb_sp, cond_neigh=emb_nb)

TIMES["total"] = time.perf_counter() - T0
params = pd.DataFrame({
    "parameter": ["patient", "depth_batch", "model", "n_cells", "n_mye",
                  "n_targets", "csf1_token", "null_genes", "batch_size",
                  "gpu", "torch"],
    "value": [PATIENT, str(meta["depth_batch"]), str(MODEL), len(ids),
              len(mye_rows), n_t, csf1_tok, ";".join(null["symbol"]), BATCH_SIZE,
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              torch.__version__]})
import pandas.io.formats.excel as pdx   # noqa: E402
pdx.ExcelFormatter.header_style = None
with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
    params.to_excel(xw, sheet_name="parameters", index=False)
    pd.DataFrame(CHECKS).to_excel(xw, sheet_name="checks", index=False)
    neg.to_excel(xw, sheet_name="negative_control_cell_emb", index=False)
    per_cell.to_excel(xw, sheet_name="per_cell", index=False)
    pd.DataFrame({"stage": list(TIMES), "seconds": list(TIMES.values())}).to_excel(
        xw, sheet_name="timing", index=False)
    pd.DataFrame({"run": list(GPU_MEM), "peak_gpu_GiB": list(GPU_MEM.values())}).to_excel(
        xw, sheet_name="gpu_memory", index=False)
failed = [c["check"] for c in CHECKS if not c["pass"]]
print(f"[done] {PATIENT}: checks failed: {failed if failed else 'none'} "
      f"({TIMES['total']:.0f} s)", flush=True)
if failed:
    sys.exit(1)
