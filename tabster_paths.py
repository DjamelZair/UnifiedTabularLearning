from __future__ import annotations
from pathlib import Path
import os

# ================================
# Root detection logic
# ================================
def _detect_root(start: Path) -> Path:
    # Try current dir, then up to 3 levels
    for cand in [start, *list(start.parents)[:3]]:
        if (cand / "datasets").exists() and (cand / "merged_data").exists():
            return cand

    # Allow override with env var
    env = os.getenv("TABSTER_PROJECT_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "datasets").exists() and (p / "merged_data").exists():
            return p

    # Fallback: assume current directory
    return start

# ================================
# Project paths
# ================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _detect_root(THIS_DIR)

# Top-level folders
DATA_DIR    = PROJECT_ROOT / "datasets"
MERGED_DIR  = PROJECT_ROOT / "merged_data"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"  # optional

# Create folders if missing
for d in [MERGED_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================================
# Merged & intermediate files
# ================================
UNIFIED_NPZ         = MERGED_DIR / "unified_data_v6.npz"
TRAINVAL_LATENTS    = MERGED_DIR / "trainval_latents_v6.npy"
TEST_LATENTS        = MERGED_DIR / "test_latents_v6.npy"
TEST_PREPROCESSED   = MERGED_DIR / "test_preprocessed_v6.npz"

ZSCALE_PARAMS       = MERGED_DIR / "zscaler_numeric_params_v6.npz"
IMPUTER_BINARY      = MERGED_DIR / "imputer_binary_v6.pkl"
FEATURE_CATALOG     = MERGED_DIR / "feature_catalog_v6.csv"
MERGE_META_JSON     = MERGED_DIR / "merge_meta_v6.json"

# ================================
# Model files
# ================================
AUTOENCODER_PATH = MODELS_DIR / "autoencoder.pt"
ENCODER_PATH     = MODELS_DIR / "encoder.pt"
MLP_MODEL_SINGLEHEAD = RESULTS_DIR / "results_singlehead/unified_mlp_singlehead_best_v6.pt"
MLP_MODEL_MULTIHEAD  = RESULTS_DIR / "results_multihead/unified_mlp_multihead_best_v7.pt"
# ================================
# Example result files
# ================================
CLF_METRICS_CSV = METRICS_DIR / "mlp_metrics.csv"
CLF_CONFMAT_PNG = PLOTS_DIR / "mlp_confusion_matrix.png"

## tabster_paths.py  (put this file at: /home/.../tabster/TABSTER2/tabster_paths.py)
#from __future__ import annotations
#import os
#from pathlib import Path
#
#def _detect_root(start: Path) -> Path:
#    # Try current dir, then climb up 3 levels and pick the first that has both folders
#    for cand in [start, *list(start.parents)[:3]]:
#        if (cand / "datasets").exists() and (cand / "merged_data").exists():
#            return cand
#    # Allow explicit override
#    env = os.getenv("TABSTER_PROJECT_ROOT")
#    if env:
#        p = Path(env).resolve()
#        if (p / "datasets").exists() and (p / "merged_data").exists():
#            return p
#    # Last resort: use current file’s directory
#    return start
#
#THIS_DIR = Path(__file__).resolve().parent
#PROJECT_ROOT = _detect_root(THIS_DIR)
#DATA_DIR    = PROJECT_ROOT / "datasets"
#MERGED_DIR  = PROJECT_ROOT / "merged_data"
#MODELS_DIR  = PROJECT_ROOT / "models"
#SCRIPTS_DIR = PROJECT_ROOT / "scripts"
#
