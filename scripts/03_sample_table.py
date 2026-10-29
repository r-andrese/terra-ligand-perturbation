#!/usr/bin/env python
"""
Step 03 (Yan Xenium) - per-patient table for TERRA sample inclusion
File: scripts/03_sample_table.py

Purpose
  Summarise every patient exported for TERRA (02_build_h5ad.py adata.h5ad
  files) and apply the pre-specified inclusion rules. Also used to choose 
  the single patient for initial testing (smallest patient that passes, 
  preferably deep and with > 1 tissue piece).

Definitions
  Fibroblast        cell_type == "Fibro" and cell_state in FIB_STATES
  CSF1+ fibroblast  fibroblast with CSF1 raw count > 0
  Target macrophage cell_type == "Mye" and cell_state in MAC_TARGET
  FIB_STATES and MAC_TARGET are read from config/analysis.cfg (SOURCE_STATES
  and TARGET_STATES).
  Depth             median nCount_Xenium over annotated cells (cell_state not NA)

Inclusion (all must hold)
  median depth >= MIN_MEDIAN_DEPTH (100)
  CSF1+ fibroblasts >= MIN_CSF1POS_FIB (50)
  target macrophages >= MIN_TARGET_MAC (50)

Depth batch
  Reported for information: "deep" if median depth >= DEEP_DEPTH (400),
  otherwise "shallow" (the cohort is bimodal, ~140 vs ~730 counts).

Reads : <WORK_DIR>/export/<PATIENT>/adata.h5ad
Writes: <RESULTS_DIR>/03_sample_table/03_sample_table.xlsx (unformatted),
        and .csv copy

Paths come from config/paths.cfg via env/terra_env.sh.
Runs inside the TERRA container (CPU only), via scripts/03_sample_table.sh.
"""

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

EXP_DIR = Path(os.environ["WORK_DIR"]) / "export"
RES_DIR = Path(os.environ["RESULTS_DIR"]) / "03_sample_table"

from analysis_config import load_states   # noqa: E402
FIB_STATES, MAC_TARGET = load_states()
LIGAND = "CSF1"

MIN_MEDIAN_DEPTH = 100
MIN_CSF1POS_FIB = 50
MIN_TARGET_MAC = 50
DEEP_DEPTH = 400

RES_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(d / "adata.h5ad" for d in EXP_DIR.iterdir() if (d / "adata.h5ad").exists())
assert files, f"no adata.h5ad files under {EXP_DIR}"
print(f"[1] {len(files)} patients", flush=True)

rows = []
for f in files:
    a = ad.read_h5ad(f)
    o = a.obs
    assert LIGAND in a.var_names, f"{f.parent.name}: {LIGAND} not on the panel"

    lig = np.asarray(a[:, LIGAND].X.todense()).ravel()
    annot = o["cell_state"].notna().to_numpy()
    fib = ((o["cell_type"] == "Fibro") & o["cell_state"].isin(FIB_STATES)).to_numpy()
    mac = ((o["cell_type"] == "Mye") & o["cell_state"].isin(MAC_TARGET)).to_numpy()
    depth = float(np.median(o.loc[annot, "nCount_Xenium"])) if annot.any() else np.nan

    row = dict(
        patient=f.parent.name,
        n_cells=a.n_obs,
        n_annotated=int(annot.sum()),
        n_pieces=o["sample"].nunique(),
        cells_per_piece=";".join(str(v) for v in o["sample"].value_counts().sort_index().to_numpy()),
        median_depth_annot=depth,
        n_fib=int(fib.sum()),
        n_csf1pos_fib=int((fib & (lig > 0)).sum()),
        n_target_mac=int(mac.sum()),
    )
    for s in FIB_STATES:
        sel = (o["cell_state"] == s).to_numpy() & fib
        row[f"n_{s}"] = int(sel.sum())
        row[f"n_csf1pos_{s}"] = int((sel & (lig > 0)).sum())
    for s in MAC_TARGET:
        row[f"n_{s}"] = int(((o["cell_state"] == s).to_numpy() & mac).sum())
    rows.append(row)
    print(f"    {row['patient']}: {row['n_cells']} cells, depth {depth:.0f}, "
          f"{row['n_csf1pos_fib']} CSF1+ fib, {row['n_target_mac']} target mac", flush=True)
    del a

T = pd.DataFrame(rows)
T["frac_csf1pos_fib"] = (T["n_csf1pos_fib"] / T["n_fib"].where(T["n_fib"] > 0)).round(3)
T["depth_batch"] = np.where(T["median_depth_annot"] >= DEEP_DEPTH, "deep", "shallow")
T["pass_depth"] = T["median_depth_annot"] >= MIN_MEDIAN_DEPTH
T["pass_csf1pos_fib"] = T["n_csf1pos_fib"] >= MIN_CSF1POS_FIB
T["pass_target_mac"] = T["n_target_mac"] >= MIN_TARGET_MAC
T["included"] = T["pass_depth"] & T["pass_csf1pos_fib"] & T["pass_target_mac"]
T = T.sort_values(["included", "n_cells"], ascending=[False, True]).reset_index(drop=True)

params = pd.DataFrame({
    "parameter": ["FIB_STATES", "MAC_TARGET", "LIGAND", "MIN_MEDIAN_DEPTH",
                  "MIN_CSF1POS_FIB", "MIN_TARGET_MAC", "DEEP_DEPTH"],
    "value": [";".join(FIB_STATES), ";".join(MAC_TARGET), LIGAND, MIN_MEDIAN_DEPTH,
              MIN_CSF1POS_FIB, MIN_TARGET_MAC, DEEP_DEPTH]})

T.to_csv(RES_DIR / "03_sample_table.csv", index=False)
try:
    with pd.ExcelWriter(RES_DIR / "03_sample_table.xlsx") as xw:
        T.to_excel(xw, sheet_name="per_patient", index=False)
        params.to_excel(xw, sheet_name="parameters", index=False)
except ModuleNotFoundError:
    print("    NOTE: openpyxl not installed in the container; wrote .csv only "
          "(it is in env/versions/requirements.lock; re-run env/install.sh)", flush=True)

show = ["patient", "n_cells", "n_pieces", "median_depth_annot", "depth_batch",
        "n_fib", "n_csf1pos_fib", "frac_csf1pos_fib", "n_target_mac", "included"]
print("\n[2] per-patient summary (included first, then by size):")
print(T[show].to_string(index=False), flush=True)
print(f"\n[3] included: {int(T['included'].sum())} / {len(T)}; "
      f"excluded: {', '.join(T.loc[~T['included'], 'patient'])}", flush=True)
print(f"    written to {RES_DIR}", flush=True)
