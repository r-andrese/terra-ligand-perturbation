"""
analysis_config.py - cell states analysed, read from config/analysis.cfg

Steps 03, 04 and 07 import this module.
Copy config/analysis.cfg.example and fill in the cell_state labels of the dataset.

  SOURCE_STATES  fibroblast states whose CSF1+ cells are the ligand source
  TARGET_STATES  one or more macrophage states whose representation is read out 
"""

from pathlib import Path

CFG = Path(__file__).resolve().parent.parent / "config" / "analysis.cfg"


def load_states():
    """Return (source_states, target_states) as lists of cell_state labels."""
    if not CFG.exists():
        raise SystemExit("%s not found; copy config/analysis.cfg.example to "
                         "config/analysis.cfg and fill in the cell states" % CFG)
    vals = {}
    for line in CFG.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = [s.strip() for s in v.split(",") if s.strip()]
    for k in ("SOURCE_STATES", "TARGET_STATES"):
        if not vals.get(k):
            raise SystemExit("%s is not set in %s" % (k, CFG))
        if any(s.startswith("<") for s in vals[k]):
            raise SystemExit("%s in %s still holds placeholders" % (k, CFG))
    if len(set(vals["TARGET_STATES"])) != len(vals["TARGET_STATES"]):
        raise SystemExit("TARGET_STATES lists a state more than once")
    return vals["SOURCE_STATES"], vals["TARGET_STATES"]
