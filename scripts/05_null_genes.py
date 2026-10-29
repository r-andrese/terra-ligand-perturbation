#!/usr/bin/env python
"""
Step 05 (Yan Xenium) - TERRA knockout: rank-matched null genes
File: scripts/05_null_genes.py

Purpose
  Choose the 20 null genes for the in silico knockout, once for the whole
  cohort. A null gene is knocked out in exactly the same neighbour blocks as
  CSF1 (the blocks of CSF1+ fibroblasts), so it has to be seen by the model
  in those blocks about as often, and at about the same rank, as CSF1.

Rules (fixed before results are seen)
  Statistics are taken over the own 256-token block of every CSF1+
  fibroblast in the 38 included patients (from 04):
    presence     fraction of CSF1+ fibroblasts whose block contains the gene
    median rank  median position (1 = highest) when present
  A gene is a candidate when
    |median rank - CSF1 median rank| <= RANK_BAND        (40 positions)
    |presence - CSF1 presence|      <= PRESENCE_BAND     (0.15)
    it is not CSF1 and not in EXCLUDED_LIGANDS (ligands with established
    direct effects on macrophage recruitment, survival or polarisation, so
    that the null does not contain real signal).
  20 genes are drawn from the candidates with seed 1234. The script stops if
  fewer than 20 candidates exist; the bands are not widened automatically.
  Also collects the per-patient 04 summaries into one cohort table
  (depth batch, cell counts, CSF1 visibility).

Reads : <KO_DIR>/<P>/cf_blocks.npz, <KO_DIR>/<P>/meta.npz
        (KO_DIR = <WORK_DIR>/terra_ko),
        <RESULTS_DIR>/04_tokenise/04_tokenise_<P>.xlsx
Writes: <RES_DIR>/05_null_genes.xlsx   (unformatted)
        <RES_DIR>/05_null_genes.tsv    (read by 06)
        (RES_DIR = <RESULTS_DIR>/05_null_genes)

Paths come from config/paths.cfg via env/terra_env.sh.
Runs inside the TERRA container (CPU only), via scripts/05_null_genes.sh.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(os.environ["WORK_DIR"]) / "export"
F_DIR = Path(os.environ["RESULTS_DIR"]) / "04_tokenise"
RES_DIR = Path(os.environ["RESULTS_DIR"]) / "05_null_genes"
KO_DIR = Path(os.environ["WORK_DIR"]) / "terra_ko"
EXCLUDED = {"P10", "P23", "P71", "P87", "P89", "P93"}
N_INCLUDED = 38
GENE = "CSF1"
N_NULL = 20
SEED = 1234
RANK_BAND = 40
PRESENCE_BAND = 0.15
EXCLUDED_LIGANDS = [
    "CSF1", "IL34", "CSF2", "CSF3", "IL3", "IL4", "IL13", "IL6", "IL10",
    "IL1A", "IL1B", "IL18", "IL33", "TNF", "IFNG", "TGFB1", "TGFB2", "TGFB3",
    "CCL2", "CCL3", "CCL4", "CCL5", "CCL7", "CCL8", "CCL13", "CCL18", "CCL19",
    "CCL21", "CCL22", "CXCL8", "CXCL9", "CXCL10", "CXCL12", "CX3CL1", "SPP1",
    "GAS6", "PROS1", "VEGFA", "VEGFC", "MIF", "C3", "C5", "HGF", "IGF1",
    "GDF15", "ANGPTL4", "PDGFB", "LGALS9", "THBS1", "CD47"]

RES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Patients and CSF1+ fibroblast blocks
# ---------------------------------------------------------------------------
pts = sorted(d.name for d in EXP_DIR.iterdir()
             if d.is_dir() and (d / "adata.h5ad").exists() and d.name not in EXCLUDED)
assert len(pts) == N_INCLUDED, f"expected {N_INCLUDED} patients, found {len(pts)}"
missing = [p for p in pts if not (KO_DIR / p / "cf_blocks.npz").exists()]
if missing:
    sys.exit(f"04 outputs missing for: {missing}")
print(f"[1] {len(pts)} patients", flush=True)

blocks, summaries = [], []
var = None
for p in pts:
    blocks.append(np.load(KO_DIR / p / "cf_blocks.npz", allow_pickle=True)["cf_tokens"])
    m = np.load(KO_DIR / p / "meta.npz", allow_pickle=True)
    v = pd.DataFrame({"symbol": m["var_symbol"].astype(str),
                      "ensembl": m["var_ensembl"].astype(str),
                      "token": m["var_token"]})
    if var is None:
        var, csf1_tok = v, int(m["csf1_token"])
    else:
        assert v.equals(var), f"{p}: harmonised gene panel differs"
    summaries.append(pd.read_excel(F_DIR / f"04_tokenise_{p}.xlsx",
                                   sheet_name="summary"))
M = np.vstack(blocks)                     # (n CSF1+ fibroblasts, 256)
n_cf = M.shape[0]
print(f"    {n_cf} CSF1+ fibroblasts in total", flush=True)

# ---------------------------------------------------------------------------
# 2. Presence and median rank of every gene in those blocks
# ---------------------------------------------------------------------------
pos = np.broadcast_to(np.arange(1, M.shape[1] + 1), M.shape)
long = pd.DataFrame({"token": M.ravel(), "rank": pos.ravel()})
long = long[long["token"] != 0]
stats = (long.groupby("token")["rank"]
         .agg(n_present="size", median_rank="median", mean_rank="mean")
         .reset_index())
stats["presence"] = stats["n_present"] / n_cf
stats = stats.merge(var, on="token", how="left")
assert stats["symbol"].notna().all(), "tokens without a panel gene"
stats = stats.sort_values("median_rank").reset_index(drop=True)

csf1 = stats[stats["symbol"] == GENE]
assert len(csf1) == 1, "CSF1 never seen in CSF1+ fibroblast blocks"
c_rank = float(csf1["median_rank"].iloc[0])
c_pres = float(csf1["presence"].iloc[0])
print(f"[2] CSF1: presence {c_pres:.3f}, median rank {c_rank:.1f}", flush=True)

# ---------------------------------------------------------------------------
# 3. Candidates and draw
# ---------------------------------------------------------------------------
stats["in_rank_band"] = (stats["median_rank"] - c_rank).abs() <= RANK_BAND
stats["in_presence_band"] = (stats["presence"] - c_pres).abs() <= PRESENCE_BAND
stats["excluded_ligand"] = stats["symbol"].isin(EXCLUDED_LIGANDS)
stats["candidate"] = (stats["in_rank_band"] & stats["in_presence_band"]
                      & ~stats["excluded_ligand"] & (stats["symbol"] != GENE))
cand = stats[stats["candidate"]].sort_values("symbol").reset_index(drop=True)
print(f"[3] {len(cand)} candidates", flush=True)
if len(cand) < N_NULL:
    sys.exit(f"only {len(cand)} candidates (< {N_NULL}); bands not widened")
rng = np.random.default_rng(SEED)
pick = np.sort(rng.choice(len(cand), size=N_NULL, replace=False))
sel = cand.iloc[pick].reset_index(drop=True)
print("    selected: " + ", ".join(sel["symbol"]), flush=True)

# ---------------------------------------------------------------------------
# 4. Write
# ---------------------------------------------------------------------------
cohort = pd.concat(summaries, ignore_index=True)
params = pd.DataFrame({
    "parameter": ["n_patients", "n_csf1pos_fib", "csf1_presence",
                  "csf1_median_rank", "rank_band", "presence_band", "n_null",
                  "seed", "n_candidates", "n_deep", "n_shallow"],
    "value": [len(pts), n_cf, c_pres, c_rank, RANK_BAND, PRESENCE_BAND, N_NULL,
              SEED, len(cand), int((cohort["depth_batch"] == "deep").sum()),
              int((cohort["depth_batch"] == "shallow").sum())]})

import pandas.io.formats.excel as pdx   # noqa: E402
pdx.ExcelFormatter.header_style = None
with pd.ExcelWriter(RES_DIR / "05_null_genes.xlsx", engine="openpyxl") as xw:
    params.to_excel(xw, sheet_name="parameters", index=False)
    sel.to_excel(xw, sheet_name="selected", index=False)
    cand.to_excel(xw, sheet_name="candidates", index=False)
    stats.to_excel(xw, sheet_name="all_genes", index=False)
    pd.DataFrame({"symbol": EXCLUDED_LIGANDS}).to_excel(
        xw, sheet_name="excluded_ligands", index=False)
    cohort.to_excel(xw, sheet_name="patients", index=False)
sel[["symbol", "ensembl", "token", "presence", "median_rank"]].to_csv(
    RES_DIR / "05_null_genes.tsv", sep="\t", index=False)
print(f"[4] written to {RES_DIR}; depth batches: "
      f"{params.loc[9, 'value']} deep / {params.loc[10, 'value']} shallow", flush=True)
