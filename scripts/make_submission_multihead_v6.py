# TABSTER2/scripts/make_submission_multihead_v6.py
# Build Kaggle-style submissions from MULTI-HEAD predictions.
# Outputs:
#   results_latents_v6/submission_multihead_v6.csv          (combined)
#   results_latents_v6/submission_multihead_v6_covtype.csv  (Covertype only)
#   results_latents_v6/submission_multihead_v6_heloc.csv    (HELOC only)
#   results_latents_v6/submission_multihead_v6_higgs.csv    (Higgs only)

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
# Preferred multi-head test predictions file; will fall back to internal-test if missing
PRED_CSV_PRIMARY   = OUTDIR / "test_predictions_multihead_v6.csv"
PRED_CSV_FALLBACK  = OUTDIR / "internal_test_predictions_multihead_v6.csv"

SUB_ALL_CSV   = OUTDIR / "submission_multihead_v6.csv"
SUB_COV_CSV   = OUTDIR / "submission_multihead_v6_covtype.csv"
SUB_HEL_CSV   = OUTDIR / "submission_multihead_v6_heloc.csv"
SUB_HIG_CSV   = OUTDIR / "submission_multihead_v6_higgs.csv"

TEST_PATHS = {
    "Covertype": DATA_DIR / "covtype_test.csv",
    "HELOC":     DATA_DIR / "heloc_test.csv",
    "Higgs":     DATA_DIR / "higgs_test.csv",
}

# Unified label blocks (same as training)
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

def to_unified(local, ds_name):
    start, _ = BLOCKS[ds_name]
    return local + start

def make_expected_ids(ds_name: str, n_rows: int) -> np.ndarray:
    start = EXPECTED[ds_name]["id_start"]
    return np.arange(start, start + n_rows, dtype=int)

def load_predictions(pred_arg: str | None = None) -> pd.DataFrame:
    if pred_arg is not None:
        path = Path(pred_arg)
        if not path.exists():
            raise FileNotFoundError(f"Provided predictions path does not exist: {path}")
        return pd.read_csv(path)

    # Default behavior: prefer test_predictions_multihead_v6.csv
    if PRED_CSV_PRIMARY.exists():
        return pd.read_csv(PRED_CSV_PRIMARY)

    # Fallback for a smoke test on internal split
    if PRED_CSV_FALLBACK.exists():
        print(f"[WARN] {PRED_CSV_PRIMARY.name} not found; using fallback {PRED_CSV_FALLBACK.name}", flush=True)
        return pd.read_csv(PRED_CSV_FALLBACK)

    raise FileNotFoundError(
        f"Missing predictions file. Looked for:\n"
        f"  {PRED_CSV_PRIMARY}\n"
        f"  {PRED_CSV_FALLBACK}\n"
        f"Optionally pass a path as CLI arg."
    )

def main():
    pred_arg = sys.argv[1] if len(sys.argv) > 1 else None
    p_header("MAKE COMBINED SUBMISSION (Multi-Head: CoverType + HELOC + Higgs)")

    preds = load_predictions(pred_arg)
    p_kv("Pred path", pred_arg if pred_arg else ("fallback" if preds is not None else "primary"))
    p_kv("Pred shape", preds.shape)
    print("[Pred head]")
    print(preds.head(5).to_string(index=False), flush=True)

    # Accept either 'y_pred_unified' (preferred) or 'y_pred_local' with conversion
    cols = set(preds.columns)
    if not {"dataset_id"}.issubset(cols):
        raise ValueError("Predictions CSV must contain dataset_id column")

    if "y_pred_unified" in cols:
        preds["y_pred_unified"] = preds["y_pred_unified"].astype(int)
    elif "y_pred_local" in cols:
        # Convert local -> unified according to dataset_id
        uni = np.empty(len(preds), dtype=int)
        ds_ids = preds["dataset_id"].to_numpy()
        yloc   = preds["y_pred_local"].to_numpy().astype(int)
        for name, (start, _) in BLOCKS.items():
            m = ds_ids == DS_INDEX[name]
            uni[m] = to_unified(yloc[m], name)
        preds["y_pred_unified"] = uni
    else:
        raise ValueError("Predictions CSV must contain either y_pred_unified or y_pred_local")

    p_counts("Pred dataset_id counts", preds["dataset_id"].to_numpy())

    per_ds_frames = []
    out_per_ds = {}

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

        ds_mask = preds["dataset_id"].to_numpy() == DS_INDEX[ds]
        pred_unified = preds.loc[ds_mask, "y_pred_unified"].to_numpy()
        if len(pred_unified) != len(df_test):
            raise RuntimeError(f"{ds} rows mismatch: preds {len(pred_unified)} vs test {len(df_test)}")

        ids = make_expected_ids(ds, len(df_test))

        local = to_local(pred_unified, ds).astype(int)
        if ds == "Covertype":
            local = local + 1  # Kaggle labels are 1..7
            if local.min() < 1 or local.max() > 7:
                raise RuntimeError(f"{ds} labels out of range after +1 shift: {local.min()}..{local.max()}")
        else:
            if not set(np.unique(local)).issubset({0, 1}):
                raise RuntimeError(f"{ds} labels not in {{0,1}}: {np.unique(local)}")

        out_ds = pd.DataFrame({"ID": ids, "Prediction": local})
        p_kv(f"{ds} out rows", len(out_ds))
        p_kv(f"{ds} ID range", f"{out_ds['ID'].min()}..{out_ds['ID'].max()}")
        p_counts(f"{ds} label counts", out_ds["Prediction"].to_numpy())

        per_ds_frames.append(out_ds)
        out_per_ds[ds] = out_ds

    p_header("CONCATENATE AND FINAL CHECKS")
    out_all = pd.concat(per_ds_frames, axis=0, ignore_index=True).sort_values("ID").reset_index(drop=True)

    id_min, id_max = out_all["ID"].min(), out_all["ID"].max()
    p_kv("Final rows", len(out_all))
    p_kv("Final ID range", f"{id_min}..{id_max}")

    dup = out_all["ID"].duplicated().sum()
    if dup > 0:
        raise RuntimeError(f"Found {dup} duplicated IDs in final submission.")

    expected_global_max = EXPECTED["Higgs"]["id_start"] + EXPECTED["Higgs"]["n_rows"] - 1
    if id_min != EXPECTED["Covertype"]["id_start"] or id_max != expected_global_max:
        raise RuntimeError(f"Global ID bounds {id_min}..{id_max} differ from expected "
                           f"{EXPECTED['Covertype']['id_start']}..{expected_global_max}")

    print("[Final head]")
    print(out_all.head(10).to_string(index=False), flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_all.to_csv(SUB_ALL_CSV, index=False)
    out_per_ds["Covertype"].to_csv(SUB_COV_CSV, index=False)
    out_per_ds["HELOC"].to_csv(SUB_HEL_CSV, index=False)
    out_per_ds["Higgs"].to_csv(SUB_HIG_CSV, index=False)

    print(f"Saved {SUB_ALL_CSV}", flush=True)
    print(f"Saved {SUB_COV_CSV}", flush=True)
    print(f"Saved {SUB_HEL_CSV}", flush=True)
    print(f"Saved {SUB_HIG_CSV}", flush=True)

if __name__ == "__main__":
    main()
