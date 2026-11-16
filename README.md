# terra-ligand-perturbation

In silico perturbation of a fibroblast-derived ligand (CSF1) in the spatial
neighbourhood of tumour macrophages, using the pretrained spatial
transcriptomics foundation model TERRA (Birk *et al.*, 2026) on the public
triple-negative breast cancer (TNBC) Xenium cohort of Yan *et al.* (2026), 
with rank-matched null-gene and positive controls.

**Version 0.1.** Packaged from the scripts of the original analysis with 
configurable cell states. TERRA is used for zero-shot inference.

**TERRA version.** Developed and run with `terra-st` 0.1.13 and TERRA-112M at
Hugging Face revision `00d1c977`. TERRA is under active development and its
interface may change on publication; later versions may need adjustments.

---

## Aim

Cancer-associated fibroblasts (CAFs) are a major source of ligands that act on 
macrophages within the tumour microenvironment. Foundation models that encode a
cell together with its spatial neighbours make it possible to ask *in silico* 
whether removing a ligand from neighbouring cells changes the model's representation 
of a receiving cell. This study used publicly available 
human triple-negative breast cancer (TNBC) Xenium data to determine whether 
removing CSF1 from CSF1-expressing fibroblast neighbours shifts the TERRA 
representation of selected macrophage states (the target states), and whether 
any observed shift is specific to CSF1.

## Data and model

* **Data.** Yan, Y. *et al.* Ecotypes of triple-negative breast cancer in
  response to chemotherapy. *Nature* **654**, 1088-1097 (2026).
  [doi:10.1038/s41586-026-10469-9](https://doi.org/10.1038/s41586-026-10469-9).
  Code: [navinlabcode/tnbc-chemo](https://github.com/navinlabcode/tnbc-chemo).
  The per-patient Xenium objects are distributed by the authors. Cell types 
  and states are the authors' annotations.
* **Model.** Birk, S. *et al.* Multi-scale modeling of human tissues from
  spatial transcriptomics with TERRA. *bioRxiv* (2026).
  [doi:10.64898/2026.07.29.741565](https://www.biorxiv.org/content/10.64898/2026.07.29.741565v1.full).
  Code: [Lotfollahi-lab/terra](https://github.com/Lotfollahi-lab/terra);
  documentation: [terra-st.readthedocs.io](https://terra-st.readthedocs.io/).
  The weights are downloaded from Hugging Face by `env/install.sh` and are
  not redistributed here; check their licence on the model card.

## Design

### Cohort and cell definitions

All 44 patients with Xenium data are exported on the gene panel shared by
all sections. Tissue pieces are defined as connected regions of occupied
100 um bins, so that spatial neighbourhoods never bridge separate pieces.
Inclusion rules are median depth >= 100 counts; >= 50 CSF1+ fibroblasts; >= 50 target macrophages, 
leaving **38 patients**.

* CSF1+ fibroblast: a fibroblast in one of the source states with a raw
  *CSF1* count > 0.
* Target macrophages: cells in the target states.
* Source and target states are cell-state labels of the dataset, set in
  `config/analysis.cfg`. 
* Depth batch: median total counts of annotated cells on the shared panel >= 500 
  ("deep", 12 patients) or below ("shallow", 26 patients), defined in
  step 04 and used by step 07. 

### Model input and perturbations

TERRA-112M represents each cell as 11 blocks of 256 gene tokens: the cell
itself, then its 10 nearest neighbours, each block listing that cell's genes
in order of normalised expression. Every patient is tokenised **once**, and
every perturbation is built by editing those tokens directly, only in the
neighbour blocks that belong to a CSF1+ fibroblast. 

Two observations from TERRA's current behaviour motivated this design:

* Setting *CSF1* counts to zero and re-tokenising changes the tokens of
  non-fibroblast cells beyond those whose CSF1+ fibroblast neighbour
  blocks contain *CSF1*. This is consistent with re-tokenisation and
  re-normalization of the edited fibroblasts.

* In `terra-st 0.1.13`, `perturb_dataset` with perturbation_target="neighborhood" 
  and CSF1-positive fibroblast identifiers also edited neighbour blocks 
  belonging to unlisted cells, in addition to the intended edits in the 
  listed fibroblasts’ blocks. Later versions may behave differently.

| Condition | Edit in each CSF1+ fibroblast neighbour block | Why it is needed |
|---|---|---|
| `CSF1_remove` (test) | *CSF1* token and value removed; remaining tokens move up | The perturbation of interest |
| `NULL_<gene>` x 20 | Same edit for a null gene | Reference distribution for an edit of comparable size at the same positions |
| `POS_block` | Whole block emptied | Positive control: shows the pipeline can detect a neighbourhood change |
| `CSF1_zero` | *CSF1* value set to 0, token kept | Checks whether the result depends on how a loss is encoded |
| `cell_emb` readout | (all conditions) | Negative control: a neighbour edit cannot change the cell's own embedding |

**Null genes** are chosen once for the cohort: genes whose presence in the
own 256-token block of CSF1+ fibroblasts and whose median rank there lie
within 0.15 and 40 positions of *CSF1* (presence 0.66, median rank 60),
excluding *CSF1* and ligands with established direct effects on
macrophages. Twenty genes are drawn from the candidates (seed 1234).

### Readouts

For each target macrophage the model returns a spatial cell embedding
(primary) and a neighbourhood embedding (secondary).

* **Classifier readout (primary).** Logistic regression on baseline
  embeddings of all macrophage states, scoring P(target) = P(any target
  state vs other macrophage states). Fitted as leave-one-donor-out. 
  For each edited cell, dP = P(perturbed) - P(baseline).
* **Magnitude readouts (descriptive).** Cosine shift per cell, and TERRA's
  own Wasserstein-2 distance (W2) between baseline and perturbed embeddings
  of the edited cells of each donor. 

### Why gene-level controls matter

Changing any input to a model changes its output, and larger or more prominent 
changes tend to produce larger effects. Therefore, reporting only the magnitude 
of the output change after removing a gene cannot determine whether that gene 
has a specific effect or whether removing a comparable gene in the same position 
would produce a similar change.

To address this, each test gene is compared with matched null genes: genes selected 
in the same way but not expected to have a particular effect. Rather than assessing 
whether removal of the test gene changes the output, the analysis evaluates whether 
its effect exceeds that of comparable genes.

A positive control (a gene expected to produce a clear effect) confirms that the 
analysis can detect real changes. If the positive control performs as expected 
while a test gene produces only a small change, the small effect can be interpreted 
as biologically or functionally weak rather than as a failure of the method.

### Limitations of the approach

* **Low-count transcripts.** A ligand is represented in the model only if it falls 
  within a cell’s 256 most highly expressed genes. 
* **Zero-shot model.** TERRA was not trained to predict responses to
  perturbation; a change in its representation is not a predicted
  biological response.
* **Readout.** The classifier readout depends on the authors' label-transfer
  annotations of macrophage states.

## Outputs

Every step writes an `.xlsx` summary. 

Step 07 produces two workbooks: 

* **S07_summary.xlsx** one table plus a sheet explaining every column.
* **S07_details.xlsx** detailed results:

| Sheet | Content |
|---|---|
| `README` | settings, definitions and the settings signature of the run |
| `wilcoxon` | the two-sided test across donors, overall and per depth batch |
| `gene_rank` | cohort median of donor mean dP for each condition, with ranks |
| `donor_d`, `donor_condition` | per-donor d and per-donor summaries for every condition |
| `strata_n_edited`, `by_target_state` | the same summaries by number of edited neighbours and by target state |
| `w2_donor`, `w2_vs_null`, `w2_rank` | TERRA's W2 distance per donor, relative to the null genes, and ranked |
| `classifier_auc` | held-out AUC of the leave-one-donor-out classifier, per donor |
| `input_qc` | per-patient structure checks of the step 06 files |

## Repository layout

```
config/paths.cfg.example     paths to edit (copy to config/paths.cfg)
config/analysis.cfg.example  cell states analysed (copy to config/analysis.cfg)
env/terra_env.sh             shared settings
env/install.sh               one-off install
env/check_config.sh          checks configuration, software, input, model 
env/versions/                pinned package list and recorded environment
scripts/01_export.R  .sh     export per-patient matrices 
scripts/02_build_h5ad        assemble one .h5ad per patient
scripts/03_sample_table      per-patient table and pre-specified inclusion
scripts/04_tokenise          tokenise each included patient once 
scripts/05_null_genes        choose the 20 rank-matched null genes
scripts/06_ko_embed          build the perturbations and embed 
scripts/07_ko_pool           classifier, readouts and summary statistics
scripts/analysis_config.py   reads the cell states from config/analysis.cfg
```

Each step writes an unformatted `.xlsx` summary to
`RESULTS_DIR/<step>/`; large intermediate files go to `WORK_DIR`. Steps
01, 02, 04 and 06 skip patients whose outputs already exist, and step 07
reuses cached per-donor predictions (`FORCE_REBUILD = False` at the top of
each script); steps 03 and 05 are quick and always recompute.

## How to run

**Requirements.** Linux cluster with Sun Grid Engine (job scripts use
`qsub`), Singularity or Apptainer, an NVIDIA GPU with >= 40 GB memory for
step 06, a node with internet access for the installation, and R (developed
with 4.4.1) with `Matrix`, `igraph` and `openxlsx` for step 01. 

```bash
git clone <this repository> && cd terra-ligand-perturbation
cp config/paths.cfg.example config/paths.cfg       # edit the five paths
cp config/analysis.cfg.example config/analysis.cfg # set the cell states
bash env/install.sh                            # once, on a node with internet
bash env/check_config.sh                       # also inside a GPU session
mkdir -p logs
```

All jobs are submitted from the repository root; `-hold_jid` chains them.

```bash
qsub -o logs/ scripts/01_export.sh
qsub -o logs/ -hold_jid tlp_01_export      scripts/02_build_h5ad.sh
qsub -o logs/ -hold_jid tlp_02_build_h5ad  scripts/03_sample_table.sh
qsub -o logs/ -hold_jid tlp_03_sample_table scripts/04_tokenise.sh   # array, 38 tasks
qsub -o logs/ -hold_jid tlp_04_tokenise    scripts/05_null_genes.sh
```

Then run one patient on the GPU and inspect its log before the full array;
the array skips that patient afterwards:

```bash
qsub -o logs/ -t 1-1 -v PATIENT=P12 scripts/06_ko_embed.sh
qsub -o logs/ scripts/06_ko_embed.sh                                # GPU array
qsub -o logs/ -hold_jid tlp_06_ko_embed scripts/07_ko_pool.sh
```

Step 06 needs a card with >= 40 GB; restrict the GPU request accordingly
(the job script requests only `gpu=true`). R must be available to the
step 01 job.

## Planned developments

Version 0.1 reproduces the analysis described above. A future version is
intended to generalise the pipeline, so that any ligand, source cell states
and target cell states can be tested on a single-cell-resolution spatial
dataset, with:

* the ligand and cell states set in the configuration;
* patient inclusion derived from the configured states and thresholds;
* summary figures of the classifier and magnitude readouts.

## How to cite

If you use this code, please cite the software:

> Andres Ejarque R. *terra-ligand-perturbation: zero-shot in silico ligand
> perturbation in spatial transcriptomics using TERRA* (version 0.1.0). 2026.
> https://github.com/r-andrese/terra-ligand-perturbation

## Author and contact

Rosa Andres Ejarque, King's College London.

Questions, bug reports and suggestions are welcome as
[GitHub issues](https://github.com/r-andrese/terra-ligand-perturbation/issues).

## Citation and licence

If you use this code, please cite Yan *et al.* (2026) and Birk *et al.*
(2026) above, and this repository. Code: MIT licence (see `LICENSE`).
TERRA weights: CC BY-NC 4.0, as stated by their authors.

