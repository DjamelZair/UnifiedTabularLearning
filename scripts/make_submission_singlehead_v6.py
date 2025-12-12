# scripts/make_submission_singlehead_v6.py
# Kaggle submission builder for SINGLE-HEAD unified MLP
# Repo root: UnifiedTabularLearning/

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Locate project root (UnifiedTabularLearning/)
# ------------------------------------------------------------------
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from tabster_paths import PROJECT_ROOT, DATA_DIR

# ------------------------------------------------------------------
# Configuration (must match training)
# ------------------------------------------------------------------
ORDER = ["Covertype", "Higgs", "HELOC"]
DS_INDEX = {name: i for i, name in enumerate(ORDER)}

BLOCKS = {
    "Covertype": (0, 7),
    "Higgs":     (7, 2),
    "HELOC":     (9, 2),
}

EXPECTED = {
    "Covertype": {"n_rows": 3500,  "id_start": 1},
    "HELOC":     {"n_rows": 1046,  "id_start": 3501},
    "Higgs":     {"n_rows": 75000, "id_start": 4547},
}

# ---- SINGLE-HEAD prediction sources ----
# Prefer real Kaggle-test predictions; fall back to internal test for smoke checks.
PRED_CSV_PRIMARY  = PROJECT_ROOT / "results" / "test_predictions_v6.csv"
PRED_CSV_FALLBACK = PROJECT_ROOT / "results" / "metrics" / "internal_test_predictions_v6.csv"

OUTDIR  = PROJECT_ROOT / "results" / "submissions"
SUB_CSV = OUTDIR / "submission_singlehead_v6.csv"

TEST_PATHS = {
    "Covertype": DATA_DIR / "covtype_test.csv",
    "HELOC":     DATA_DIR / "heloc_test.csv",
    "Higgs":     DATA_DIR / "higgs_test.csv",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def to_local(y_unified, ds):
    start, _ = BLOCKS[ds]
    return y_unified - start

def make_ids(ds, n):
    start = EXPECTED[ds]["id_start"]
    return np.arange(start, start + n, dtype=int)

def log_counts(label, arr):
    u, c = np.unique(arr, return_counts=True)
    print(label + ": " + ", ".join(f"{int(a)}:{int(b)}" for a, b in zip(u, c)))

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("\n" + "=" * 80)
    print("KAGGLE SUBMISSION — SINGLE-HEAD MLP (UnifiedTabularLearning)")
    print("=" * 80)

    pred_path = PRED_CSV_PRIMARY if PRED_CSV_PRIMARY.exists() else PRED_CSV_FALLBACK
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file. Looked for:\n"
            f"  {PRED_CSV_PRIMARY}\n"
            f"  {PRED_CSV_FALLBACK}"
        )

    preds = pd.read_csv(pred_path)
    print(f"Using predictions from: {pred_path}")

    required = {"dataset_id", "y_pred_unified"}
    if not required.issubset(preds.columns):
        raise ValueError(f"Expected columns {required}, got {list(preds.columns)}")

    log_counts("dataset_id distribution", preds["dataset_id"].values)

    frames = []

    for ds in ["Covertype", "HELOC", "Higgs"]:
        print(f"\nProcessing {ds}")

        test_df = pd.read_csv(TEST_PATHS[ds])
        mask = preds["dataset_id"].values == DS_INDEX[ds]
        y_u = preds.loc[mask, "y_pred_unified"].to_numpy()

        if len(y_u) != len(test_df):
            raise RuntimeError(f"{ds}: {len(y_u)} preds vs {len(test_df)} test rows")

        start, length = BLOCKS[ds]
        # Force predictions into the valid block for this dataset
        clipped = np.clip(y_u, start, start + length - 1)
        if not np.array_equal(clipped, y_u):
            n_bad = int((clipped != y_u).sum())
            print(f"[WARN] {ds}: {n_bad} predictions outside block {start}-{start+length-1}; clipped.")
        y_local = (clipped - start).astype(int)

        if ds == "Covertype":
            y_local += 1  # Kaggle expects 1–7
        else:
            assert set(np.unique(y_local)).issubset({0, 1})

        log_counts(f"{ds} labels", y_local)

        frames.append(
            pd.DataFrame({
                "ID": make_ids(ds, len(test_df)),
                "Prediction": y_local
            })
        )

    submission = (
        pd.concat(frames)
        .sort_values("ID")
        .reset_index(drop=True)
    )

    if submission["ID"].duplicated().any():
        raise RuntimeError("Duplicate IDs in submission")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUB_CSV, index=False)

    print("\nSaved submission to:")
    print(SUB_CSV)
    print(submission.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
