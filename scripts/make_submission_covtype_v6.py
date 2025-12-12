# TABSTER2/scripts/make_submission_covtype_v6.py
# Build Kaggle-style submission for CoverType ONLY.
# IDs: 1..3500; labels: 1..7 (converted from unified 0..6).

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np
import pandas as pd
from tabster_paths import PROJECT_ROOT, DATA_DIR

ORDER_DEFAULT = ["Covertype", "Higgs", "HELOC"]
DS_INDEX = {name: i for i, name in enumerate(ORDER_DEFAULT)}

OUTDIR = PROJECT_ROOT / "results_latents_v6"
PRED_CSV = OUTDIR / "test_predictions_v6.csv"
SUB_CSV  = OUTDIR / "submission_covtype_v6.csv"
TEST_PATH = DATA_DIR / "covtype_test.csv"

BLOCKS = {"Covertype": (0, 7), "Higgs": (7, 2), "HELOC": (9, 2)}

def p_header(t):
    print("\n" + "="*80 + f"\n{t}\n" + "="*80, flush=True)
def p_kv(k,v): print(f"[{k}] {v}", flush=True)
def p_counts(label, arr):
    arr = np.asarray(arr); u,c = np.unique(arr, return_counts=True)
    print(f"{label} -> " + ", ".join(f"{int(ui)}:{int(ci)}" for ui,ci in zip(u,c)), flush=True)

def to_local(unified, ds_name):
    start, _ = BLOCKS[ds_name]
    return unified - start

def main():
    p_header("COVERTYPE SUBMISSION")
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions: {PRED_CSV}")
    preds = pd.read_csv(PRED_CSV)
    need = {"dataset_id", "y_pred_unified"}
    if not need.issubset(preds.columns):
        raise ValueError(f"Predictions CSV must contain {need}, got {list(preds.columns)}")
    p_kv("Pred shape", preds.shape)
    p_counts("Pred dataset_id counts", preds["dataset_id"].to_numpy())

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing covtype test: {TEST_PATH}")
    df_test = pd.read_csv(TEST_PATH)
    exp_n = 3500
    if len(df_test) != exp_n:
        print(f"[WARN] covtype test rows {len(df_test)} != expected {exp_n}", flush=True)

    mask = preds["dataset_id"].to_numpy() == DS_INDEX["Covertype"]
    pred_u = preds.loc[mask, "y_pred_unified"].to_numpy()
    if len(pred_u) != len(df_test):
        raise RuntimeError(f"Row mismatch: preds {len(pred_u)} vs test {len(df_test)}")

    # unified 0..6 -> local 1..7
    local = to_local(pred_u, "Covertype").astype(int) + 1
    if local.min() < 1 or local.max() > 7:
        raise RuntimeError(f"CoverType labels out of range 1..7: {local.min()}..{local.max()}")

    ids = np.arange(1, 1 + len(df_test), dtype=int)
    out = pd.DataFrame({"ID": ids, "Prediction": local}).sort_values("ID")
    # integrity
    if out["ID"].duplicated().any():
        raise RuntimeError("Duplicate IDs in covtype submission")
    p_kv("Rows", len(out))
    p_kv("ID range", f"{out['ID'].min()}..{out['ID'].max()}")
    p_counts("Label counts", out["Prediction"].to_numpy())

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUB_CSV, index=False)
    print(f"Saved {SUB_CSV}", flush=True)

if __name__ == "__main__":
    main()
