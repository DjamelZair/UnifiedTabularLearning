# TABSTER2/scripts/make_submission_v6.py
# Build ONE Kaggle-style submission for ALL datasets (CoverType, HELOC, Higgs).
# Enforces canonical contiguous ID ranges:
#   CoverType: 1..3500
#   HELOC:     3501..4546
#   Higgs:     4547..79546
# Converts unified predictions to local labels per dataset and aligns counts.

import sys
from pathlib import Path

# allow importing tabster_paths
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np
import pandas as pd
from tabster_paths import PROJECT_ROOT, DATA_DIR

# Canonical order used during training/inference for dataset_id
ORDER_DEFAULT = ["Covertype", "Higgs", "HELOC"]
DS_INDEX = {name: i for i, name in enumerate(ORDER_DEFAULT)}

EXPECTED = {
    "Covertype": {"n_rows": 3500,  "id_start": 1},
    "HELOC":     {"n_rows": 1046,  "id_start": 3501},
    "Higgs":     {"n_rows": 75000, "id_start": 4547},
}

OUTDIR = PROJECT_ROOT / "results_latents_v6"
PRED_CSV = OUTDIR / "test_predictions_v6.csv"
SUB_CSV  = OUTDIR / "submission_v6.csv"

TEST_PATHS = {
    "Covertype": DATA_DIR / "covtype_test.csv",
    "HELOC":     DATA_DIR / "heloc_test.csv",
    "Higgs":     DATA_DIR / "higgs_test.csv",
}

BLOCKS = {"Covertype": (0, 7), "Higgs": (7, 2), "HELOC": (9, 2)}

def p_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80, flush=True)

def p_kv(k, v):
    print(f"[{k}] {v}", flush=True)

def p_counts(label, arr):
    arr = np.asarray(arr)
    uniq, cnt = np.unique(arr, return_counts=True)
    pairs = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt)])
    print(f"{label} -> {pairs}", flush=True)

def to_local(unified, ds_name):
    start, _ = BLOCKS[ds_name]
    return unified - start

def make_expected_ids(ds_name: str, n_rows: int) -> np.ndarray:
    start = EXPECTED[ds_name]["id_start"]
    return np.arange(start, start + n_rows, dtype=int)

def main():
    p_header("MAKE COMBINED SUBMISSION (CoverType + HELOC + Higgs)")
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_CSV}")
    preds = pd.read_csv(PRED_CSV)
    p_kv("Pred file", str(PRED_CSV))
    p_kv("Pred shape", preds.shape)
    print("[Pred head]")
    print(preds.head(5).to_string(index=False), flush=True)

    need = {"dataset_id", "y_pred_unified"}
    if not need.issubset(set(preds.columns)):
        raise ValueError(f"Predictions CSV must contain {need}, got {list(preds.columns)}")

    p_counts("Pred dataset_id counts", preds["dataset_id"].to_numpy())

    per_ds_frames = []
    total_expected = 0

    # Process in global ID order: Covertype -> HELOC -> Higgs
    for ds in ["Covertype", "HELOC", "Higgs"]:
        p_header(f"PROCESS {ds}")
        test_path = TEST_PATHS[ds]
        if not test_path.exists():
            raise FileNotFoundError(f"Missing {ds} test CSV: {test_path}")
        df_test = pd.read_csv(test_path)
        exp_n = EXPECTED[ds]["n_rows"]
        if len(df_test) != exp_n:
            print(f"[WARN] {ds} test rows {len(df_test)} != expected {exp_n}. Proceeding.", flush=True)

        # predictions for this dataset
        ds_mask = preds["dataset_id"].to_numpy() == DS_INDEX[ds]
        pred_unified = preds.loc[ds_mask, "y_pred_unified"].to_numpy()
        if len(pred_unified) != len(df_test):
            raise RuntimeError(f"{ds} rows mismatch: preds {len(pred_unified)} vs test {len(df_test)}")

        # expected IDs (always synthesized, we ignore any file ID columns)
        ids = make_expected_ids(ds, len(df_test))

        # local label conversion
        local = to_local(pred_unified, ds).astype(int)
        if ds == "Covertype":
            # Kaggle labels are 1..7
            local = local + 1
            if local.min() < 1 or local.max() > 7:
                raise RuntimeError(f"{ds} labels out of range after +1 shift: {local.min()}..{local.max()}")
        elif ds in ("HELOC", "Higgs"):
            if not set(np.unique(local)).issubset({0, 1}):
                raise RuntimeError(f"{ds} labels not in {{0,1}}: {np.unique(local)}")

        out_ds = pd.DataFrame({"ID": ids, "Prediction": local})
        p_kv(f"{ds} out rows", len(out_ds))
        p_kv(f"{ds} ID range", f"{out_ds['ID'].min()}..{out_ds['ID'].max()}")
        p_counts(f"{ds} label counts", out_ds["Prediction"].to_numpy())

        per_ds_frames.append(out_ds)
        total_expected += len(out_ds)

    p_header("CONCATENATE AND FINAL CHECKS")
    out = pd.concat(per_ds_frames, axis=0, ignore_index=True)
    out = out.sort_values("ID").reset_index(drop=True)

    id_min, id_max = out["ID"].min(), out["ID"].max()
    p_kv("Final rows", len(out))
    p_kv("Final ID range", f"{id_min}..{id_max}")

    # Strict integrity: duplicates and bounds
    dup = out["ID"].duplicated().sum()
    if dup > 0:
        raise RuntimeError(f"Found {dup} duplicated IDs in final submission.")

    expected_global_max = EXPECTED["Higgs"]["id_start"] + EXPECTED["Higgs"]["n_rows"] - 1
    if id_min != EXPECTED["Covertype"]["id_start"] or id_max != expected_global_max:
        raise RuntimeError(f"Global ID bounds {id_min}..{id_max} differ from expected "
                           f"{EXPECTED['Covertype']['id_start']}..{expected_global_max}")

    print("[Final head]")
    print(out.head(10).to_string(index=False), flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUB_CSV, index=False)
    print(f"Saved {SUB_CSV}", flush=True)

if __name__ == "__main__":
    main()
