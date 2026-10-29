#!/usr/bin/env Rscript
# =============================================================================
# Step 01 (Yan Xenium) - export sections for TERRA
# File: scripts/01_export.R
#
# Purpose
#   Write every Yan Xenium patient section to a plain, language-neutral format
#   that the Python step (02_build_h5ad.py) assembles into one .h5ad per
#   patient for TERRA. 
#
# Selection rules 
#   Genes   - the intersection of the gene panels of all patients, in the order
#             of the first patient's panel, so every section has the same genes.
#   Tissue  - connected tissue pieces: 100 um bins, >= 5 cells per bin, rook
#             connectivity, pieces < 10 bins = debris. Debris cells and cells
#             in sub-threshold bins are dropped. Pieces are numbered 1..k within
#             a patient by decreasing cell number. TERRA's sample identifier is
#             patient x piece (e.g. "P11_t1"), so the kNN graph is built within a
#             piece.
#   Cells   - all cells with >= MIN_GENES_CELL (10) genes detected on the shared
#             panel, annotated or not. This matches TERRA's pretraining QC
#             (cells with fewer than 10 expressed genes removed). Unannotated
#             cells are kept because they are real neighbours.
#
# Reads   : <INPUT_DIR>/<PATIENT>/mat_counts.rds, df_meta.rds, spatial_xy.rds
# Writes  : <WORK_DIR>/export/genes_shared.tsv
#           <WORK_DIR>/export/<PATIENT>/x.bin, i.bin, p.bin (sparse counts, see below)
#           <WORK_DIR>/export/<PATIENT>/cells.csv    (one row per exported cell)
#           <WORK_DIR>/export/<PATIENT>/manifest.txt (shapes and checksums)
#           <RESULTS_DIR>/01_export/01_export.xlsx  (per-patient summary)
#
# Paths come from config/paths.cfg via env/terra_env.sh. Runs on the host via 
# scripts/01_export.sh.
#
# Sparse format
#   Compressed sparse column (CSC) of the genes x cells matrix, i.e. the slots
#   of an R dgCMatrix: x = counts (int32), i = 0-based gene index of each value
#   (int32), p = column pointers (int32, length n_cells + 1). All little-endian.
#   The Python step reads these with numpy.fromfile and checks them against
#   the manifest (n_cells, n_genes, nnz, total_counts).
#
# Notes
#   - FORCE_REBUILD = FALSE skips patients whose manifest already exists.
# =============================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(igraph)
  library(openxlsx)
})

# -- Paths (from config/paths.cfg via env/terra_env.sh) -----------------------
env_path <- function(k) {
  v <- Sys.getenv(k)
  if (!nzchar(v)) stop(sprintf("%s is not set; run via scripts/01_export.sh", k))
  v
}
BASE    <- env_path("INPUT_DIR")
EXP_DIR <- file.path(env_path("WORK_DIR"), "export")
RES_DIR <- file.path(env_path("RESULTS_DIR"), "01_export")

# Tissue piece definition
BIN_UM          <- 100
MIN_CELLS_BIN   <- 5
CONNECTIVITY    <- 4      # rook
MIN_REGION_BINS <- 10     # pieces below this (0.1 mm2) are debris

MIN_GENES_CELL  <- 10     # TERRA pretraining QC: >= 10 expressed genes

FORCE_REBUILD   <- FALSE  # TRUE re-exports patients that already have a manifest

dir.create(EXP_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(RES_DIR, showWarnings = FALSE, recursive = TRUE)

# -- Tissue pieces ------------------------------------------------------------
# Returns one integer per cell: 0 = debris / sub-threshold bin, otherwise the
# piece id (arbitrary order; renumbered by size in the main loop).
tissue_components <- function(x, y) {
  bx <- floor(x / BIN_UM); by <- floor(y / BIN_UM)
  bkey <- paste(bx, by, sep = "_")
  tb  <- table(bkey)
  occ <- names(tb)[tb >= MIN_CELLS_BIN]
  if (length(occ) == 0) return(rep(0L, length(x)))

  obx <- as.integer(sub("_.*$", "", occ))
  oby <- as.integer(sub("^.*_", "", occ))
  n <- length(occ); idx <- seq_len(n); names(idx) <- occ

  if (CONNECTIVITY == 4) { dx <- c(1, 0); dy <- c(0, 1) }
  else                   { dx <- c(1, 0, 1, 1); dy <- c(0, 1, 1, -1) }

  ei <- integer(0); ej <- integer(0)
  for (k in seq_along(dx)) {
    nb  <- paste(obx + dx[k], oby + dy[k], sep = "_")
    hit <- match(nb, occ); ok <- !is.na(hit)
    ei <- c(ei, idx[ok]); ej <- c(ej, hit[ok])
  }
  g <- igraph::make_empty_graph(n = n, directed = FALSE)
  if (length(ei) > 0) g <- igraph::add_edges(g, rbind(ei, ej))
  memb <- as.integer(igraph::components(g)$membership)

  sz <- table(memb)
  small <- as.integer(names(sz)[sz < MIN_REGION_BINS])
  memb[memb %in% small] <- 0L

  out <- memb[match(bkey, occ)]
  out[is.na(out)] <- 0L
  out
}

# -- Write one integer vector as little-endian int32 --------------------------
write_int32 <- function(v, path) {
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(as.integer(v), con, size = 4L, endian = "little")
}

# -- Pass 1: shared gene panel ------------------------------------------------
pats <- list.dirs(BASE, recursive = FALSE, full.names = FALSE)
message(sprintf("[1] %d patient directories; reading gene panels", length(pats)))

panels <- vector("list", length(pats)); names(panels) <- pats
for (p in pats) {
  panels[[p]] <- rownames(readRDS(file.path(BASE, p, "mat_counts.rds")))
  gc(verbose = FALSE)
}
shared_set <- Reduce(intersect, panels)
shared     <- panels[[1]][panels[[1]] %in% shared_set]
stopifnot(length(shared) > 0, !anyDuplicated(shared))
writeLines(shared, file.path(EXP_DIR, "genes_shared.tsv"))
message(sprintf("    shared panel: %d genes (panel sizes %s)", length(shared),
                paste(sort(unique(lengths(panels))), collapse = " / ")))

# -- Pass 2: export each patient ----------------------------------------------
rows <- list()

for (p in pats) {
  out_p <- file.path(EXP_DIR, p)
  if (!FORCE_REBUILD && file.exists(file.path(out_p, "manifest.txt"))) {
    message(sprintf("    %s already exported, skipped", p))
    next
  }
  dir.create(out_p, showWarnings = FALSE, recursive = TRUE)

  md  <- readRDS(file.path(BASE, p, "df_meta.rds"))
  xy  <- readRDS(file.path(BASE, p, "spatial_xy.rds"))
  mat <- readRDS(file.path(BASE, p, "mat_counts.rds"))

  # alignment of the three objects (row order must be identical)
  stopifnot(nrow(xy) == nrow(md), identical(rownames(xy), rownames(md)),
            ncol(mat) == nrow(md))
  if (!is.null(colnames(mat)) && !identical(colnames(mat), rownames(md)))
    stop(sprintf("%s: colnames(mat_counts) do not match rownames(df_meta)", p))

  n_in <- nrow(md)
  x <- as.numeric(xy$x); y <- as.numeric(xy$y)
  ok <- is.finite(x) & is.finite(y)

  # tissue pieces on all cells with valid coordinates
  comp <- rep(0L, n_in)
  comp[ok] <- tissue_components(x[ok], y[ok])
  sizes <- sort(table(comp[comp > 0]), decreasing = TRUE)
  piece <- match(as.character(comp), names(sizes))
  piece[is.na(piece)] <- 0L

  # genes detected per cell on the shared panel
  m <- drop0(as(mat[shared, , drop = FALSE], "CsparseMatrix"))
  n_genes_shared <- diff(m@p)

  keep <- ok & piece > 0L & n_genes_shared >= MIN_GENES_CELL
  m <- m[, keep, drop = FALSE]

  # counts must be integers; abort rather than silently round
  if (length(m@x) && any(abs(m@x - round(m@x)) > 1e-6))
    stop(sprintf("%s: non-integer values in mat_counts", p))

  write_int32(round(m@x), file.path(out_p, "x.bin"))
  write_int32(m@i,        file.path(out_p, "i.bin"))
  write_int32(m@p,        file.path(out_p, "p.bin"))

  cells <- data.frame(
    cell_id          = rownames(md)[keep],
    patient          = p,
    piece            = piece[keep],
    sample_id        = sprintf("%s_t%d", p, piece[keep]),
    x                = x[keep],
    y                = y[keep],
    cell_type        = as.character(md$cell_type)[keep],
    cell_state       = as.character(md$cell_state)[keep],
    Niche            = as.character(md$Niche)[keep],
    paper_patient_id = as.character(md$paper_patient_id)[keep],
    nCount_Xenium    = as.numeric(md$nCount_Xenium)[keep],
    nFeature_Xenium  = as.numeric(md$nFeature_Xenium)[keep],
    n_genes_shared   = n_genes_shared[keep],
    stringsAsFactors = FALSE
  )
  write.csv(cells, file.path(out_p, "cells.csv"), row.names = FALSE)

  writeLines(c(
    sprintf("patient=%s",      p),
    sprintf("n_cells=%d",      ncol(m)),
    sprintf("n_genes=%d",      nrow(m)),
    sprintf("nnz=%d",          length(m@x)),
    sprintf("total_counts=%.0f", sum(m@x)),
    "layout=CSC genes x cells; x,i,p int32 little-endian; i 0-based",
    "genes=../genes_shared.tsv"
  ), file.path(out_p, "manifest.txt"))

  rows[[p]] <- data.frame(
    patient              = p,
    n_cells_in           = n_in,
    n_bad_xy             = sum(!ok),
    n_debris             = sum(ok & piece == 0L),
    n_low_genes          = sum(ok & piece > 0L & n_genes_shared < MIN_GENES_CELL),
    n_cells_exported     = ncol(m),
    n_annotated_exported = sum(!is.na(cells$cell_state)),
    n_pieces             = length(unique(cells$piece)),
    cells_per_piece      = paste(as.integer(table(cells$piece)), collapse = ";"),
    panel_size_original  = nrow(mat),
    nnz                  = length(m@x),
    stringsAsFactors     = FALSE
  )
  message(sprintf("    %s: %d -> %d cells (%d debris, %d < %d genes), %d piece(s)",
                  p, n_in, ncol(m), rows[[p]]$n_debris, rows[[p]]$n_low_genes,
                  MIN_GENES_CELL, rows[[p]]$n_pieces))

  rm(md, xy, mat, m, cells); gc(verbose = FALSE)
}

# -- Summary ------------------------------------------------------------------
if (length(rows) > 0) {
  S <- do.call(rbind, rows)
  message(sprintf("\n[2] Exported %d patients, %d cells in total",
                  nrow(S), sum(S$n_cells_exported)))
  wb <- createWorkbook()
  addWorksheet(wb, "per_patient");  writeData(wb, "per_patient", S)
  addWorksheet(wb, "parameters")
  writeData(wb, "parameters", data.frame(
    parameter = c("BIN_UM", "MIN_CELLS_BIN", "CONNECTIVITY", "MIN_REGION_BINS",
                  "MIN_GENES_CELL", "n_genes_shared"),
    value     = c(BIN_UM, MIN_CELLS_BIN, CONNECTIVITY, MIN_REGION_BINS,
                  MIN_GENES_CELL, length(shared))))
  saveWorkbook(wb, file.path(RES_DIR, "01_export.xlsx"), overwrite = TRUE)
}

message(sprintf("\n[3] Export written to %s", EXP_DIR))
sessionInfo()
