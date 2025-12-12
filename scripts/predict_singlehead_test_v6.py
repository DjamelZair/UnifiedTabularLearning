# scripts/predict_singlehead_and_submit_v6.py
# Use SINGLE-HEAD MLP on TEST latents and build Kaggle submission.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Make sure we can import tabster_paths
# ---------------------------------------------------------------------
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from tabster_paths import (
    MERGED_DIR,
    RESULTS_DIR,
    TEST_PREPROCESSED,
    TEST_LATENTS,
)

# We hard-code the single-head checkpoint to avoid the multi-head override in tabster_paths
SINGLEHEAD_CKPT = RESULTS_DIR / "results_singlehead" / "unified_mlp_singlehead_best_v6.pt"

# Unified label space
N_CLASSES = 11
ORDER = ["Covertype", "Higgs", "HELOC"]          # dataset_ids: 0,1,2 respectively

# Kaggle ID layout and expected rows
EXPECTED = {
    "Covertype": {"n_rows": 3500,  "id_start": 1},
    "HELOC":     {"n_rows": 1046,  "id_start": 3501},
    "Higgs":     {"n_rows": 75000, "id_start": 4547},
}

# Unified blocks per dataset
# Covertype: 0..6 (7 classes)
# Higgs    : 7..8 (2 classes)
# HELOC    : 9..10 (2 classes)
BLOCKS = {
    "Covertype": (0, 7),
    "Higgs":     (7, 2),
    "HELOC":     (9, 2),
}

OUT_PRED_CSV = RESULTS_DIR / "test_predictions_v6.csv"
OUT_SUB_CSV  = RESULTS_DIR / "submission_singlehead_v6.csv"


def p(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------
# Model definitions (must match training)
# ---------------------------------------------------------------------
class Trunk(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SingleHead(nn.Module):
    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, h):
        return self.out(h)


# ---------------------------------------------------------------------
def to_local(unified: np.ndarray, ds: str) -> np.ndarray:
    """
    Map unified label indices back to dataset-local labels.
    Covertype: 0..6  -> local 0..6  (we will +1 for Kaggle)
    Higgs    : 7..8  -> local 0..1
    HELOC    : 9..10 -> local 0..1
    """
    start, length = BLOCKS[ds]
    u = unified.astype(int)
    local = u - start
    if not np.all((local >= 0) & (local < length)):
        bad = u[(local < 0) | (local >= length)]
        raise ValueError(f"Unified labels {bad} out of block for {ds} (start={start}, len={length})")
    return local


def load_test_latents_and_ids():
    assert TEST_LATENTS.exists(), f"Missing TEST latents: {TEST_LATENTS}"
    assert TEST_PREPROCESSED.exists(), f"Missing TEST preprocessed NPZ: {TEST_PREPROCESSED}"

    Z = np.load(TEST_LATENTS).astype(np.float32)
    npz = np.load(TEST_PREPROCESSED)
    ds_ids = npz["dataset_ids"].astype(int)

    if len(Z) != len(ds_ids):
        raise RuntimeError(f"Latents vs dataset_ids length mismatch: {len(Z)} vs {len(ds_ids)}")

    p(f"Loaded TEST latents: {Z.shape}")
    p(f"Loaded dataset_ids : {np.bincount(ds_ids, minlength=len(ORDER))}")

    return Z, ds_ids


def load_singlehead_model(d_in: int):
    assert SINGLEHEAD_CKPT.exists(), f"Missing single-head checkpoint: {SINGLEHEAD_CKPT}"
    state = torch.load(SINGLEHEAD_CKPT, map_location="cpu")

    cfg = state.get("config", {})
    hidden = int(cfg.get("HIDDEN", 256))
    dropout = float(cfg.get("DROPOUT", 0.0))

    trunk = Trunk(d_in=d_in, hidden=hidden, dropout=dropout)
    head = SingleHead(hidden=hidden, n_classes=N_CLASSES)

    trunk.load_state_dict(state["trunk"])
    head.load_state_dict(state["head"])

    trunk.eval()
    head.eval()

    p(f"Loaded SINGLE-HEAD MLP (hidden={hidden}, dropout={dropout})")

    return trunk, head


def run_inference(Z: np.ndarray, ds_ids: np.ndarray):
    d_in = Z.shape[1]
    trunk, head = load_singlehead_model(d_in)

    with torch.no_grad():
        X = torch.from_numpy(Z)
        logits = head(trunk(X))
        y_pred_unified = logits.argmax(dim=1).cpu().numpy().astype(int)

    p("Unified label distribution:")
    unique, counts = np.unique(y_pred_unified, return_counts=True)
    for u, c in zip(unique, counts):
        p(f"  class {u}: {c}")

    # Save raw predictions (for debugging)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_pred = pd.DataFrame({
        "dataset_id": ds_ids.astype(int),
        "y_pred_unified": y_pred_unified.astype(int),
    })
    df_pred.to_csv(OUT_PRED_CSV, index=False)
    p(f"Saved raw predictions → {OUT_PRED_CSV}")

    return y_pred_unified


def build_submission(y_pred_unified: np.ndarray, ds_ids: np.ndarray):
    # Basic sanity check: dataset counts
    ds_unique, ds_counts = np.unique(ds_ids, return_counts=True)
    ds_count_map = dict(zip(ds_unique, ds_counts))
    p("dataset_id distribution in TEST:")
    for i, name in enumerate(ORDER):
        p(f"  {i} ({name}): {ds_count_map.get(i, 0)}")

    frames = []

    # 1) Covertype
    ds = "Covertype"
    ds_idx = ORDER.index(ds)
    mask = (ds_ids == ds_idx)
    preds_ds = y_pred_unified[mask]
    exp = EXPECTED[ds]["n_rows"]
    if len(preds_ds) != exp:
        raise RuntimeError(f"{ds}: expected {exp} preds, got {len(preds_ds)}")

    local = to_local(preds_ds, ds)  # 0..6
    local = local + 1               # Kaggle expects 1..7
    ids = np.arange(EXPECTED[ds]["id_start"], EXPECTED[ds]["id_start"] + exp)
    frames.append(pd.DataFrame({"ID": ids, "Prediction": local.astype(int)}))

    # 2) HELOC
    ds = "HELOC"
    ds_idx = ORDER.index(ds)
    mask = (ds_ids == ds_idx)
    preds_ds = y_pred_unified[mask]
    exp = EXPECTED[ds]["n_rows"]
    if len(preds_ds) != exp:
        raise RuntimeError(f"{ds}: expected {exp} preds, got {len(preds_ds)}")

    local = to_local(preds_ds, ds)  # 0..1
    ids = np.arange(EXPECTED[ds]["id_start"], EXPECTED[ds]["id_start"] + exp)
    frames.append(pd.DataFrame({"ID": ids, "Prediction": local.astype(int)}))

    # 3) Higgs
    ds = "Higgs"
    ds_idx = ORDER.index(ds)
    mask = (ds_ids == ds_idx)
    preds_ds = y_pred_unified[mask]
    exp = EXPECTED[ds]["n_rows"]
    if len(preds_ds) != exp:
        raise RuntimeError(f"{ds}: expected {exp} preds, got {len(preds_ds)}")

    local = to_local(preds_ds, ds)  # 0..1
    ids = np.arange(EXPECTED[ds]["id_start"], EXPECTED[ds]["id_start"] + exp)
    frames.append(pd.DataFrame({"ID": ids, "Prediction": local.astype(int)}))

    sub = pd.concat(frames, axis=0, ignore_index=True)
    sub = sub.sort_values("ID").reset_index(drop=True)

    # final sanity checks
    if sub["ID"].nunique() != len(sub):
        raise RuntimeError("Duplicate IDs detected in submission")
    if sub["ID"].min() != 1 or sub["ID"].max() != 79546:
        p("[WARN] ID range is not 1..79546 – please double-check EXPECTED layout")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_SUB_CSV, index=False)
    p(f"Saved Kaggle submission → {OUT_SUB_CSV}")
    p(sub.head().to_string(index=False))
    p(sub.tail().to_string(index=False))


def main():
    print("=" * 80)
    print("PREDICT TEST + MAKE SUBMISSION — SINGLE-HEAD MLP (v6)")
    print("=" * 80)

    Z, ds_ids = load_test_latents_and_ids()
    y_pred_unified = run_inference(Z, ds_ids)
    build_submission(y_pred_unified, ds_ids)


if __name__ == "__main__":
    main()
