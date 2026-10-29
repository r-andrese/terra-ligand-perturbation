#!/usr/bin/env python
"""
Step 02 (Yan Xenium) - assemble the TERRA input .h5ad files
File: scripts/02_build_h5ad.py

Purpose
  Read the per-patient export written by 01_export.R and write
  one AnnData file per patient in the layout TERRA expects:
    X                 raw integer counts, cells x genes (CSR, float32 storage)
    var_names         gene symbols (TERRA maps them to Ensembl IDs)
    obs               cell metadata from cells.csv; obs['sample'] = patient x
                      tissue piece, the identifier used to build the kNN graph
    obs_names         "<patient>_<cell_id>": Xenium cell ids repeat across
                      patients, and TERRA's perturbation step selects cells by
                      id, so ids must be unique across the whole cohort. The
                      original id is kept in obs['cell_id'].
    obsm['spatial']   x, y in micrometres

Reads : <WORK_DIR>/export/genes_shared.tsv, <EXP_DIR>/<PATIENT>/{x,i,p}.bin,
        cells.csv, manifest.txt   (EXP_DIR = <WORK_DIR>/export)
Writes: <EXP_DIR>/<PATIENT>/adata.h5ad   (one file per patient)

FORCE_REBUILD = False skips patients whose adata.h5ad already exists.
Paths come from config/paths.cfg via env/terra_env.sh.
Runs inside the TERRA container (CPU only), via scripts/02_build_h5ad.sh.
"""

import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Same location as EXP_DIR in 01_export.R
EXP_DIR = Path(os.environ["WORK_DIR"]) / "export"
FORCE_REBUILD = False


def read_manifest(path):
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


assert EXP_DIR.is_dir(), f"export directory not found: {EXP_DIR}"
genes = pd.read_csv(EXP_DIR / "genes_shared.tsv", header=None)[0].astype(str).to_numpy()
assert len(genes) == len(set(genes)), "duplicated gene symbols in genes_shared.tsv"
print(f"[1] shared panel: {len(genes)} genes", flush=True)

patient_dirs = sorted(d for d in EXP_DIR.iterdir() if d.is_dir() and (d / "manifest.txt").exists())
assert patient_dirs, f"no exported patients (manifest.txt) under {EXP_DIR}"
print(f"[2] {len(patient_dirs)} exported patients", flush=True)

n_written = 0
for d in patient_dirs:
    out = d / "adata.h5ad"
    if out.exists() and not FORCE_REBUILD:
        print(f"    {d.name}: adata.h5ad exists, skipped", flush=True)
        continue
    t0 = time.time()

    man = read_manifest(d / "manifest.txt")
    n_cells, n_genes, nnz = int(man["n_cells"]), int(man["n_genes"]), int(man["nnz"])
    assert man["patient"] == d.name, f"{d.name}: manifest patient field is {man['patient']}"
    assert n_genes == len(genes), f"{d.name}: gene count differs from genes_shared.tsv"

    x = np.fromfile(d / "x.bin", dtype="<i4")
    i = np.fromfile(d / "i.bin", dtype="<i4")
    p = np.fromfile(d / "p.bin", dtype="<i4")
    assert len(x) == nnz and len(i) == nnz, f"{d.name}: nnz mismatch"
    assert len(p) == n_cells + 1 and p[0] == 0 and p[-1] == nnz, f"{d.name}: column pointer mismatch"
    assert (x > 0).all(), f"{d.name}: zero or negative stored counts"
    assert i.min() >= 0 and i.max() < n_genes, f"{d.name}: gene index out of range"
    assert int(x.astype(np.int64).sum()) == int(float(man["total_counts"])), f"{d.name}: total counts mismatch"

    # genes x cells CSC -> cells x genes CSR (the transpose of a CSC is a CSR)
    X = sp.csc_matrix((x.astype(np.float32), i, p), shape=(n_genes, n_cells)).T.tocsr()
    del x, i, p

    obs = pd.read_csv(d / "cells.csv", dtype={"cell_id": str})
    assert len(obs) == n_cells, f"{d.name}: cells.csv rows differ from n_cells"
    assert obs["cell_id"].is_unique, f"{d.name}: duplicated cell ids"
    assert (obs["patient"].astype(str) == d.name).all(), f"{d.name}: patient column mismatch"
    # genes per cell from the matrix must equal what the R step recorded
    assert np.array_equal(np.diff(X.indptr), obs["n_genes_shared"].to_numpy()), \
        f"{d.name}: genes-per-cell differ from cells.csv (row order mismatch?)"
    assert np.isfinite(obs[["x", "y"]].to_numpy()).all(), f"{d.name}: non-finite coordinates"

    obs.index = pd.Index((d.name + "_" + obs["cell_id"]).to_numpy(), name=None)
    obs["sample"] = obs["sample_id"].astype(str)
    for col in ["patient", "sample_id", "sample", "cell_type", "cell_state", "Niche", "paper_patient_id"]:
        obs[col] = obs[col].astype("category")

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name=None)))
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float64)

    adata.write_h5ad(out, compression="gzip")
    n_written += 1
    print(f"    {d.name}: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['sample'].nunique()} piece(s), "
          f"{adata.obs['cell_state'].notna().sum()} annotated -> {out.name} "
          f"({time.time() - t0:.0f} s)", flush=True)
    del adata, X, obs

print(f"[3] done: {n_written} written, {len(patient_dirs) - n_written} skipped", flush=True)
