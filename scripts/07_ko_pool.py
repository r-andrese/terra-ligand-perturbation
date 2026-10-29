#!/usr/bin/env python
"""
Step 07 (Yan Xenium) - TERRA in silico CSF1 knockout, cohort pooling
File: scripts/07_ko_pool.py

Purpose
  Pool the per-patient knockout embeddings from 06, score every target
  macrophage with a leave-one-donor-out classifier trained on baseline
  embeddings, and summarise the CSF1 knockout against the 20 rank-matched
  null-gene knockouts and the controls.

Inputs (one file per patient, from 06)
  <KO_DIR>/<P>/emb_ko.npz with keys
    patient, depth_batch                     scalars
    mye_cell_id, mye_state                   all myeloid cells (n_mye)
    base_mye_spatial, base_mye_neigh         n_mye x D baseline embeddings
    base_mye_cell                            not used here
    t_in_mye                                 target rows within the mye arrays
    t_cell_id, t_state, n_cf_nb              target macrophages (n_t)
    base_t_spatial, base_t_neigh             n_t x D baseline embeddings
    cond_names                               CSF1_remove, CSF1_zero, POS_block,
                                             20 x NULL_<gene>
    n_edit                                   n_t x n_cond, edited neighbours
    cond_spatial, cond_neigh                 n_cond x n_t x D

Classifier (trained on baseline embeddings of mac-* cells only)
  target    P(target): any target state vs all other mac-* states
  (target states are read from config/analysis.cfg)
  On spatial_cell_emb ("spatial", primary) and neighborhood_emb
  ("neigh", secondary). StandardScaler + L2 logistic regression with
  balanced class weights. Leave-one-donor-out: the model that scores a
  donor never saw that donor's cells. Held-out AUC is reported per donor on
  that donor's baseline mac-* cells (NA if one class is absent).

Per cell
  dP     = P(condition) - P(baseline), target macrophages with n_edit > 0
           for that condition only
  cosd   = 1 - cos(condition embedding, baseline embedding), same cells

Per donor x condition x embedding (descriptive magnitude, TERRA's metric)
  W2     = TERRA's population distance between the baseline and condition
           embeddings of the same edited target macrophages: rows
           L2-normalised, GeomLoss Sinkhorn p=2, blur 0.01, tensorized
           backend, via terra.inference.token_distance.
           _geomloss_distance_pointcloud (the function behind
           summarize_w2_by_label). No subsampling. NA when a condition
           edits no target in that donor.
  W2 is magnitude only (it cannot tell a shift towards the target states from
  one away from it), so it is never used as evidence of direction. It is
  reported raw, as W2(condition) - median W2(null), and as the rank of
  CSF1_remove's cohort median W2 among the 21 genes (1 = largest).

Summary statistics (primary readout = target classifier, spatial embedding;
no automatic call is made: see "How to read the results" in the README)
  Eligible donor : >= MIN_EDITED target macrophages edited by CSF1_remove
  d (per donor)  : mean dP(condition) - median over the 20 null genes of
                   mean dP(null); a null gene with no edited target cell in
                   that donor contributes nothing to the median (n_null_used
                   is reported)
  Donor test     : two-sided Wilcoxon signed-rank test of d = 0 over all
                   eligible donors; repeated within each depth batch
  Gene rank      : cohort median (over eligible donors) of donor mean dP for
                   each condition, ranked among the 21 genes (CSF1 + 20
                   nulls; 1 = most negative)
  Also reported  : the neighbourhood embedding, CSF1_zero and POS_block (same
                   eligible donors, same null median), magnitudes (cosd),
                   strata by edited neighbours (1, 2, 3+) and by target state

Reads : <KO_DIR>/<P>/emb_ko.npz              (KO_DIR = <WORK_DIR>/terra_ko)
Writes: <RESULTS_DIR>/07_ko_pool/07_summary.xlsx  (unformatted; one table)
        <RESULTS_DIR>/07_ko_pool/07_details.xlsx  (unformatted; all sheets)
        <WORK_DIR>/ko_pool_cache/<P>.npz      cached per-donor predictions

FORCE_REBUILD = False reuses cached per-donor predictions when the cohort
(donor list) and classifier settings are unchanged.
Needs the TERRA "perturb" extra (geomloss) for W2; the script stops at
start-up if it is missing.
Paths come from config/paths.cfg via env/terra_env.sh.
CPU only; runs inside the TERRA container via scripts/07_ko_pool.sh.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path

KO_DIR = Path(os.environ["WORK_DIR"]) / "terra_ko"
RES_DIR = Path(os.environ["RESULTS_DIR"]) / "07_ko_pool"
CACHE_DIR = Path(os.environ["WORK_DIR"]) / "ko_pool_cache"
OUT_SUMMARY = RES_DIR / "07_summary.xlsx"
OUT_DETAILS = RES_DIR / "07_details.xlsx"

FORCE_REBUILD = False
EXPECTED_N_DONORS = 38
NPROC = int(os.environ.get("NSLOTS", "1"))

# Analysis parameters
TEST_COND = "CSF1_remove"
DESCR_CONDS = ["CSF1_zero", "POS_block"]
NULL_PREFIX = "NULL_"
N_NULL = 20
MIN_EDITED = 20     # eligible donor: >= this many targets edited by TEST_COND
from analysis_config import load_states   # noqa: E402
TARGET_STATES = load_states()[1]
MAC_PREFIX = "mac-"

# Classifier settings
LR_C = 1.0
LR_MAX_ITER = 5000
LR_TOL = 1e-4

# W2 settings (TERRA defaults in summarize_w2_by_label)
W2_P = 2
W2_BLUR = 0.01
W2_BACKEND = "tensorized"
W2_MIN_CELLS = 1

CLASSIFIERS = ["target"]
EMBEDDINGS = ["spatial", "neigh"]
PRIMARY = ("target", "spatial")
STRATA = ["1", "2", "3+"]

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, str(NPROC))

import numpy as np
import pandas as pd
try:
    from scipy.stats import wilcoxon
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from openpyxl import Workbook
except ImportError as e:
    sys.exit("Missing package in this environment: %s" % e)
try:
    import geomloss  # noqa: F401  (needed by the TERRA W2 helper)
    from terra.inference.token_distance import _geomloss_distance_pointcloud
except ImportError as e:
    sys.exit("W2 needs TERRA with the 'perturb' extra (geomloss): %s" % e)

RES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def say(msg):
    print("[%7.1f s] %s" % (time.time() - T0, msg), flush=True)


def as_str(a):
    return np.asarray(a).astype(str)


def scalar(a):
    a = np.asarray(a)
    return a.item() if a.ndim == 0 else a.ravel()[0]


# ---------------------------------------------------------------------------
# 1. Discover donors and check inputs
# ---------------------------------------------------------------------------
say("1. Discovering per-patient knockout files in %s" % KO_DIR)
files = sorted(KO_DIR.glob("*/emb_ko.npz"))
donors = [f.parent.name for f in files]
if len(donors) != EXPECTED_N_DONORS:
    sys.exit("Expected %d patients, found %d: %s"
             % (EXPECTED_N_DONORS, len(donors), ", ".join(donors)))
say("   %d patients found" % len(donors))

REQ = ["patient", "depth_batch", "mye_cell_id", "mye_state",
       "base_mye_spatial", "base_mye_neigh", "t_in_mye", "t_cell_id",
       "t_state", "n_cf_nb", "base_t_spatial", "base_t_neigh",
       "cond_names", "n_edit", "cond_spatial", "cond_neigh"]


def load_npz(p, keys=None):
    # 06 stores some string keys as object arrays, which need allow_pickle;
    # these are our own files. Strings are converted with as_str() on use.
    z = np.load(KO_DIR / p / "emb_ko.npz", allow_pickle=True)
    miss = [k for k in REQ if k not in z.files]
    if miss:
        sys.exit("%s: missing keys %s" % (p, miss))
    return {k: z[k] for k in (keys or z.files)}


def target_index(t_in_mye, n_mye):
    t = np.asarray(t_in_mye)
    if t.dtype == bool:
        if t.size != n_mye:
            sys.exit("t_in_mye is boolean but its length differs from n_mye")
        return np.flatnonzero(t)
    return t.astype(np.int64)


# ---------------------------------------------------------------------------
# 2. Load baseline mac-* embeddings for classifier training, and QC
# ---------------------------------------------------------------------------
say("2. Loading baseline mac-* embeddings and checking per-patient structure")
base = {e: [] for e in EMBEDDINGS}
state_all, donor_all = [], []
meta, qc_rows = {}, []
cond_ref = None
for p in donors:
    z = load_npz(p, [k for k in REQ if not k.startswith("cond_") or k == "cond_names"])
    pat = str(scalar(z["patient"]))
    if pat != p:
        sys.exit("%s: folder name and 'patient' key disagree (%s)" % (p, pat))
    cn = list(as_str(z["cond_names"]))
    if cond_ref is None:
        cond_ref = cn
    elif cn != cond_ref:
        sys.exit("%s: cond_names differ from the first patient" % p)
    mst = as_str(z["mye_state"])
    n_mye = mst.size
    ti = target_index(z["t_in_mye"], n_mye)
    tst = as_str(z["t_state"])
    n_t = tst.size
    ne = np.asarray(z["n_edit"])
    if ne.shape != (n_t, len(cn)):
        sys.exit("%s: n_edit shape %s, expected (%d, %d)" % (p, ne.shape, n_t, len(cn)))
    chk = {}
    for e, kb, kt in (("spatial", "base_mye_spatial", "base_t_spatial"),
                      ("neigh", "base_mye_neigh", "base_t_neigh")):
        bm, bt = np.asarray(z[kb]), np.asarray(z[kt])
        if bm.shape[0] != n_mye or bt.shape[0] != n_t:
            sys.exit("%s: %s/%s row counts do not match states" % (p, kb, kt))
        chk[e] = float(np.max(np.abs(bm[ti] - bt))) if n_t else np.nan
    if not np.array_equal(mst[ti], tst):
        sys.exit("%s: mye_state[t_in_mye] differs from t_state" % p)
    is_mac = np.char.startswith(mst, MAC_PREFIX)
    for e, kb in (("spatial", "base_mye_spatial"), ("neigh", "base_mye_neigh")):
        base[e].append(np.asarray(z[kb])[is_mac].astype(np.float32))
    state_all.append(mst[is_mac])
    donor_all.append(np.full(is_mac.sum(), p))
    ncf = np.asarray(z["n_cf_nb"]).ravel()
    i_test = cn.index(TEST_COND) if TEST_COND in cn else None
    meta[p] = {"depth_batch": str(scalar(z["depth_batch"])), "n_t": n_t}
    qc_rows.append({
        "patient": p, "depth_batch": meta[p]["depth_batch"],
        "n_myeloid": n_mye, "n_mac": int(is_mac.sum()), "n_target": n_t,
        **{"n_target_state_%d" % (k + 1): int((tst == s).sum())
           for k, s in enumerate(TARGET_STATES)},
        "n_target_other_state": int((~np.isin(tst, TARGET_STATES)).sum()),
        "n_target_cf_nb_gt0": int((ncf > 0).sum()),
        "frac_nedit_CSF1_remove_eq_n_cf_nb":
            float(np.mean(ne[:, i_test] == ncf)) if (i_test is not None and n_t) else np.nan,
        "maxabs_base_t_vs_mye_spatial": chk["spatial"],
        "maxabs_base_t_vs_mye_neigh": chk["neigh"],
        "dim_spatial": int(np.asarray(z["base_t_spatial"]).shape[1]),
        "dim_neigh": int(np.asarray(z["base_t_neigh"]).shape[1]),
    })
    del z

null_names = [c for c in cond_ref if c.startswith(NULL_PREFIX)]
for c in [TEST_COND] + DESCR_CONDS:
    if c not in cond_ref:
        sys.exit("Condition %s missing from cond_names" % c)
if len(null_names) != N_NULL:
    sys.exit("Expected %d null conditions, found %d" % (N_NULL, len(null_names)))
RANK_SET = [TEST_COND] + null_names
for e in EMBEDDINGS:
    base[e] = np.vstack(base[e])
state_all = np.concatenate(state_all)
donor_all = np.concatenate(donor_all)
say("   training matrices: %s" % ", ".join(
    "%s %d x %d (%.2f GB)" % (e, base[e].shape[0], base[e].shape[1], base[e].nbytes / 1e9)
    for e in EMBEDDINGS))
say("   %d mac-* training cells; states: %s" % (
    state_all.size,
    ", ".join("%s=%d" % (s, n) for s, n in zip(*np.unique(state_all, return_counts=True)))))
bad = [r["patient"] for r in qc_rows if r["n_target_other_state"] > 0]
if bad:
    say("   WARNING: targets outside %s in %s" % (TARGET_STATES, ", ".join(bad)))


def clf_labels(clf, st):
    """Training mask and binary labels for the classifier, given mac-* states."""
    return np.ones(st.size, bool), np.isin(st, TARGET_STATES).astype(int)


# ---------------------------------------------------------------------------
# 3. Leave-one-donor-out classifiers and per-donor predictions (cached)
# ---------------------------------------------------------------------------
SETTINGS = {"donors": donors, "C": LR_C, "max_iter": LR_MAX_ITER, "tol": LR_TOL,
            "target": TARGET_STATES,
            "conds": cond_ref,
            "w2": [W2_P, W2_BLUR, W2_BACKEND, W2_MIN_CELLS]}
SIG = hashlib.md5(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()
say("3. Leave-one-donor-out fits (settings signature %s, FORCE_REBUILD=%s)"
    % (SIG[:10], FORCE_REBUILD))


def cos_dist(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return 1.0 - np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)


def l2n(x, eps=1e-12):
    """Row L2-normalisation as in TERRA's _l2_normalize_rows."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def fit_predict(p):
    z = load_npz(p, ["base_t_spatial", "base_t_neigh", "cond_spatial", "cond_neigh",
                     "n_edit", "t_state", "t_cell_id"])
    out = {"sig": np.array(SIG)}
    tr_donor = donor_all != p
    for e in EMBEDDINGS:
        bt = np.asarray(z["base_t_" + e], dtype=np.float32)
        cond = np.asarray(z["cond_" + e], dtype=np.float32)
        nc, nt, dd = cond.shape
        if nt != bt.shape[0] or nc != len(cond_ref):
            sys.exit("%s: cond_%s shape %s does not match targets/conditions"
                     % (p, e, cond.shape))
        out["cos_" + e] = cos_dist(cond, bt[None, :, :]).astype(np.float32)
        ne_p = np.asarray(z["n_edit"])
        w2 = np.full(nc, np.nan)
        for j in range(nc):
            ed = ne_p[:, j] > 0
            if ed.sum() >= W2_MIN_CELLS:
                w2[j] = _geomloss_distance_pointcloud(
                    l2n(bt[ed]), l2n(cond[j][ed]), loss="sinkhorn", p=W2_P,
                    blur=W2_BLUR, backend=W2_BACKEND, device=None)
        out["w2_" + e] = w2
        for clf in CLASSIFIERS:
            m_all, y_all = clf_labels(clf, state_all)
            tr = tr_donor & m_all
            ho = (~tr_donor) & m_all
            sc = StandardScaler().fit(base[e][tr])
            lr = LogisticRegression(C=LR_C, class_weight="balanced",
                                    max_iter=LR_MAX_ITER, tol=LR_TOL)
            lr.fit(sc.transform(base[e][tr]), y_all[tr])
            k = "%s_%s" % (clf, e)
            yh = y_all[ho]
            ph = lr.predict_proba(sc.transform(base[e][ho]))[:, 1] if ho.any() else np.array([])
            auc = roc_auc_score(yh, ph) if (yh.size and 0 < yh.sum() < yh.size) else np.nan
            out["auc_" + k] = np.array([auc, yh.size, yh.sum(), tr.sum(),
                                        int(lr.n_iter_[0]), int(lr.n_iter_[0] >= LR_MAX_ITER)])
            out["pb_" + k] = lr.predict_proba(sc.transform(bt))[:, 1].astype(np.float32) \
                if nt else np.zeros(0, np.float32)
            pc = lr.predict_proba(sc.transform(cond.reshape(nc * nt, dd)))[:, 1] \
                if nt else np.zeros(0)
            out["pc_" + k] = pc.reshape(nc, nt).astype(np.float32)
    out["n_edit"] = np.asarray(z["n_edit"]).astype(np.int32)
    out["t_state"] = as_str(z["t_state"])
    out["t_cell_id"] = as_str(z["t_cell_id"])
    return out


pred = {}
for i, p in enumerate(donors, 1):
    cf = CACHE_DIR / ("%s.npz" % p)
    if cf.exists() and not FORCE_REBUILD:
        c = dict(np.load(cf, allow_pickle=False))
        if str(c.get("sig", "")) == SIG:
            pred[p] = c
            say("   [%2d/%d] %s cached" % (i, len(donors), p))
            continue
        say("   [%2d/%d] %s cache stale (settings or cohort changed)" % (i, len(donors), p))
    pred[p] = fit_predict(p)
    np.savez(cf, **pred[p])
    say("   [%2d/%d] %s fitted, n_target=%d" % (i, len(donors), p, meta[p]["n_t"]))

# ---------------------------------------------------------------------------
# 4. Per-donor summaries
# ---------------------------------------------------------------------------
say("4. Per-donor summaries of dP and cosine shift")
ci = {c: j for j, c in enumerate(cond_ref)}
gm_rows, str_rows, st_rows, auc_rows, w2_rows = [], [], [], [], []


def summ(dp, cs):
    return {"n_edited": int(dp.size),
            "mean_dP": float(dp.mean()) if dp.size else np.nan,
            "median_dP": float(np.median(dp)) if dp.size else np.nan,
            "mean_cos_shift": float(cs.mean()) if cs.size else np.nan}


for p in donors:
    r = pred[p]
    ne = r["n_edit"]
    tst = r["t_state"]
    for e in EMBEDDINGS:
        for c in cond_ref:
            w2_rows.append({"patient": p, "depth_batch": meta[p]["depth_batch"],
                            "embedding": e, "condition": c,
                            "n_edited": int((ne[:, ci[c]] > 0).sum()),
                            "W2": float(r["w2_" + e][ci[c]])})
    for clf in CLASSIFIERS:
        for e in EMBEDDINGS:
            k = "%s_%s" % (clf, e)
            a = r["auc_" + k]
            auc_rows.append({"patient": p, "depth_batch": meta[p]["depth_batch"],
                             "classifier": clf, "embedding": e, "heldout_AUC": a[0],
                             "n_heldout": int(a[1]), "n_heldout_pos": int(a[2]),
                             "n_train": int(a[3]), "lbfgs_iter": int(a[4]),
                             "hit_max_iter": int(a[5])})
            pb, pc, cs_all = r["pb_" + k], r["pc_" + k], r["cos_" + e]
            for c in cond_ref:
                j = ci[c]
                ed = ne[:, j] > 0
                dp_all = pc[j] - pb
                base_row = {"patient": p, "depth_batch": meta[p]["depth_batch"],
                            "classifier": clf, "embedding": e, "condition": c}
                gm_rows.append({**base_row, **summ(dp_all[ed], cs_all[j][ed])})
                nej = ne[:, j]
                for s, msk in (("1", nej == 1), ("2", nej == 2), ("3+", nej >= 3)):
                    str_rows.append({**base_row, "n_edited_neighbours": s,
                                     **summ(dp_all[msk], cs_all[j][msk])})
                for s in np.unique(tst):
                    msk = ed & (tst == s)
                    st_rows.append({**base_row, "target_state": s,
                                    **summ(dp_all[msk], cs_all[j][msk])})

gm = pd.DataFrame(gm_rows)
strata = pd.DataFrame(str_rows)
bystate = pd.DataFrame(st_rows)
aucs = pd.DataFrame(auc_rows)

# ---------------------------------------------------------------------------
# 5. Donor-level d, tests and gene ranks
# ---------------------------------------------------------------------------
say("5. Donor-level statistics")
n_csf1 = gm[(gm.condition == TEST_COND) & (gm.classifier == PRIMARY[0])
            & (gm.embedding == PRIMARY[1])].set_index("patient")["n_edited"]
eligible = [p for p in donors if n_csf1[p] >= MIN_EDITED]
say("   %d of %d donors eligible (>= %d targets edited by %s)"
    % (len(eligible), len(donors), MIN_EDITED, TEST_COND))

d_rows, w_rows, rank_rows = [], [], []
batches = sorted({meta[p]["depth_batch"] for p in donors})

for clf in CLASSIFIERS:
    for e in EMBEDDINGS:
        is_primary = (clf, e) == PRIMARY
        sub = gm[(gm.classifier == clf) & (gm.embedding == e)]
        mean_tab = sub.pivot(index="patient", columns="condition", values="mean_dP")
        null_tab = mean_tab[null_names]
        null_med = null_tab.median(axis=1, skipna=True)
        n_null = null_tab.notna().sum(axis=1)
        for c in [TEST_COND] + DESCR_CONDS:
            dvals = mean_tab[c] - null_med
            for p in donors:
                d_rows.append({
                    "classifier": clf, "embedding": e, "primary_readout": int(is_primary),
                    "condition": c, "patient": p, "depth_batch": meta[p]["depth_batch"],
                    "eligible": int(p in eligible),
                    "n_edited_CSF1_remove": int(n_csf1[p]),
                    "n_edited_condition": int(sub[(sub.patient == p) & (sub.condition == c)]
                                              ["n_edited"].iloc[0]),
                    "mean_dP_condition": mean_tab.loc[p, c],
                    "median_null_mean_dP": null_med[p], "n_null_used": int(n_null[p]),
                    "d": dvals[p]})
            for grp in ["all"] + batches:
                ps = [p for p in eligible if grp == "all" or meta[p]["depth_batch"] == grp]
                x = dvals.loc[ps].dropna().values
                if x.size >= 1 and np.any(x != 0):
                    res = wilcoxon(x, alternative="two-sided")
                    stat, pv = float(res.statistic), float(res.pvalue)
                else:
                    stat, pv = np.nan, np.nan
                w_rows.append({
                    "classifier": clf, "embedding": e, "primary_readout": int(is_primary),
                    "condition": c, "depth_batch": grp, "n_donors": int(x.size),
                    "n_d_negative": int((x < 0).sum()),
                    "median_d": float(np.median(x)) if x.size else np.nan,
                    "mean_d": float(np.mean(x)) if x.size else np.nan,
                    "wilcoxon_V": stat, "p_two_sided": pv})
        cm = mean_tab.loc[eligible].median(axis=0, skipna=True)
        rk = cm[RANK_SET].rank(method="min", ascending=True)
        for c in cond_ref:
            rank_rows.append({
                "classifier": clf, "embedding": e, "primary_readout": int(is_primary),
                "condition": c, "in_rank_set": int(c in RANK_SET),
                "cohort_median_mean_dP": cm[c],
                "n_donors_nonNA": int(mean_tab.loc[eligible, c].notna().sum()),
                "rank_most_negative_of_21": int(rk[c]) if c in RANK_SET else np.nan})


w2 = pd.DataFrame(w2_rows)
w2d_rows, w2r_rows = [], []
for e in EMBEDDINGS:
    tab = w2[w2.embedding == e].pivot(index="patient", columns="condition", values="W2")
    nmed = tab[null_names].median(axis=1, skipna=True)
    nn = tab[null_names].notna().sum(axis=1)
    for c in [TEST_COND] + DESCR_CONDS:
        for p in donors:
            w2d_rows.append({"embedding": e, "condition": c, "patient": p,
                             "depth_batch": meta[p]["depth_batch"],
                             "eligible": int(p in eligible), "W2_condition": tab.loc[p, c],
                             "median_null_W2": nmed[p], "n_null_used": int(nn[p]),
                             "W2_minus_null_median": tab.loc[p, c] - nmed[p]})
    cmw = tab.loc[eligible].median(axis=0, skipna=True)
    rkw = cmw[RANK_SET].rank(method="min", ascending=False)
    for c in cond_ref:
        w2r_rows.append({"embedding": e, "condition": c, "in_rank_set": int(c in RANK_SET),
                         "cohort_median_W2": cmw[c],
                         "n_donors_nonNA": int(tab.loc[eligible, c].notna().sum()),
                         "rank_largest_of_21": int(rkw[c]) if c in RANK_SET else np.nan})
# ---------------------------------------------------------------------------
# Summary table: one row per embedding and condition
# ---------------------------------------------------------------------------
W = pd.DataFrame(w_rows)
W = W[W.depth_batch == "all"].set_index(["embedding", "condition"])
RK = pd.DataFrame(rank_rows).set_index(["embedding", "condition"])
W2R = pd.DataFrame(w2r_rows).set_index(["embedding", "condition"])
AUC = aucs.groupby("embedding")["heldout_AUC"].median()
order = [TEST_COND] + DESCR_CONDS + null_names
sum_rows = []
for e in EMBEDDINGS:
    for c in order:
        has_d = (e, c) in W.index
        sum_rows.append({
            "embedding": e, "primary_embedding": int(e == PRIMARY[1]),
            "condition": c,
            "role": "test" if c == TEST_COND else
                    "null gene" if c in null_names else "control",
            "n_donors": int(W.loc[(e, c), "n_donors"]) if has_d else np.nan,
            "median_d": W.loc[(e, c), "median_d"] if has_d else np.nan,
            "n_d_negative": int(W.loc[(e, c), "n_d_negative"]) if has_d else np.nan,
            "p_two_sided": W.loc[(e, c), "p_two_sided"] if has_d else np.nan,
            "cohort_median_mean_dP": RK.loc[(e, c), "cohort_median_mean_dP"],
            "rank_most_negative_of_21": RK.loc[(e, c), "rank_most_negative_of_21"],
            "cohort_median_W2": W2R.loc[(e, c), "cohort_median_W2"],
            "W2_rank_largest_of_21": W2R.loc[(e, c), "rank_largest_of_21"],
            "median_heldout_AUC": AUC.get(e, np.nan)})
summary = pd.DataFrame(sum_rows)
columns_doc = pd.DataFrame({"column": [
    "embedding", "primary_embedding", "condition", "role", "n_donors", "median_d",
    "n_d_negative", "p_two_sided", "cohort_median_mean_dP",
    "rank_most_negative_of_21", "cohort_median_W2", "W2_rank_largest_of_21",
    "median_heldout_AUC"], "meaning": [
    "spatial (spatial_cell_emb) or neigh (neighborhood_emb)",
    "1 for the primary embedding (spatial)",
    "%s (test), %s (controls) or NULL_<gene> (rank-matched null genes)"
    % (TEST_COND, ", ".join(DESCR_CONDS)),
    "test, control or null gene",
    "eligible donors with a value of d (>= %d targets edited by %s)" % (MIN_EDITED, TEST_COND),
    "median over donors of d = mean dP(condition) - median over the null genes of mean dP(null)",
    "donors with d < 0 (shift away from the target states beyond the null median)",
    "two-sided Wilcoxon signed-rank test of d = 0 across eligible donors",
    "median over eligible donors of the donor mean dP = P(target | perturbed) - P(target | baseline)",
    "rank of cohort_median_mean_dP among the test gene and the 20 null genes (1 = most negative)",
    "median over eligible donors of TERRA's W2 distance between baseline and perturbed embeddings",
    "rank of cohort_median_W2 among the test gene and the 20 null genes (1 = largest); magnitude only",
    "median over donors of the held-out AUC of the leave-one-donor-out classifier"]})
ps = summary[(summary.primary_embedding == 1) & (summary.condition == TEST_COND)].iloc[0]
say("   %s, %s embedding: median d = %.3g, p (two-sided) = %.3g, rank %s of 21"
    % (TEST_COND, PRIMARY[1], ps.median_d, ps.p_two_sided, ps.rank_most_negative_of_21))

# ---------------------------------------------------------------------------
# 6. Write xlsx
# ---------------------------------------------------------------------------
say("6. Writing %s and %s" % (OUT_SUMMARY, OUT_DETAILS))
readme = pd.DataFrame({"item": [
    "script", "date", "n_donors", "n_eligible", "primary_readout", "test_condition",
    "null_conditions", "min_edited_targets", "dP",
    "cos_shift", "d", "donor_test", "gene_rank",
    "classifier", "training_cells", "W2", "settings_signature"],
    "value": [
    "07_ko_pool.py", time.strftime("%Y-%m-%d %H:%M"), len(donors), len(eligible),
    "%s classifier on %s embedding" % PRIMARY, TEST_COND, ", ".join(null_names),
    MIN_EDITED,
    "P(condition) - P(baseline), target macrophages with n_edit > 0 for that condition",
    "1 - cosine(condition, baseline) embedding, same cells",
    "mean dP(condition) - median over nulls of mean dP(null); nulls with no edited target omitted",
    "two-sided Wilcoxon signed-rank test of d = 0, eligible donors; repeated per depth batch",
    "cohort median of donor mean dP, ranked among the test gene and the 20 nulls (1 = most negative)",
    "StandardScaler + L2 logistic regression, class_weight balanced, C=%g, lbfgs" % LR_C,
    "baseline embeddings of mac-* cells; leave-one-donor-out",
    "descriptive only; TERRA Sinkhorn p=%d blur=%g %s on L2-normalised rows, "
    "baseline vs condition embeddings of edited targets per donor; rank 1 = largest" % (
        W2_P, W2_BLUR, W2_BACKEND),
    SIG]})

summary_sheets = [("summary", summary), ("columns", columns_doc), ("settings", readme)]
sheets = [("settings", readme), ("wilcoxon", pd.DataFrame(w_rows)),
          ("gene_rank", pd.DataFrame(rank_rows)), ("donor_d", pd.DataFrame(d_rows)),
          ("donor_condition", gm), ("strata_n_edited", strata),
          ("by_target_state", bystate), ("w2_donor", w2),
          ("w2_vs_null", pd.DataFrame(w2d_rows)), ("w2_rank", pd.DataFrame(w2r_rows)),
          ("classifier_auc", aucs),
          ("input_qc", pd.DataFrame(qc_rows))]


def cell(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


for out, shs in ((OUT_SUMMARY, summary_sheets), (OUT_DETAILS, sheets)):
    wb = Workbook(write_only=True)
    for name, df in shs:
        ws = wb.create_sheet(name)
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append([cell(v) for v in row])
    wb.save(out)
say("Done.")
