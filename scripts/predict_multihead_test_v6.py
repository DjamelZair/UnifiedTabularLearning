# scripts/predict_multihead_test_v6.py
# Run inference on Kaggle test latents using the trained multi-head MLP (v7 ckpt).

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Import tabster_paths
# ------------------------------------------------------------------
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from tabster_paths import (
    TEST_LATENTS,
    MERGED_DIR,
    RESULTS_DIR,
)

# Multi-head checkpoint (trained v7 model)
CKPT_PATH = RESULTS_DIR / "results_multihead" / "unified_mlp_multihead_best_v7.pt"
OUT_CSV = RESULTS_DIR / "metrics" / "test_predictions_multihead_v6.csv"

# ------------------------------------------------------------------
# Model defs (match training)
# ------------------------------------------------------------------
class Trunk(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Heads(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.cov = nn.Linear(hidden, 7)
        self.hig = nn.Linear(hidden, 2)
        self.hel = nn.Linear(hidden, 2)
    def forward(self, h):
        return {"Covertype": self.cov(h), "Higgs": self.hig(h), "HELOC": self.hel(h)}

def main():
    assert TEST_LATENTS.exists(), f"Missing test latents: {TEST_LATENTS}"
    meta_npz = MERGED_DIR / "test_preprocessed_v6.npz"
    assert meta_npz.exists(), f"Missing test meta npz: {meta_npz}"
    assert CKPT_PATH.exists(), f"Missing multi-head checkpoint: {CKPT_PATH}"

    # Load test latents and meta (dataset_ids + order)
    Z = np.load(TEST_LATENTS).astype(np.float32)
    meta = np.load(meta_npz, allow_pickle=True)
    ds_ids = meta["dataset_ids"].astype(int)
    order = [str(x) for x in meta["order"]]
    assert len(ds_ids) == len(Z), f"dataset_ids len {len(ds_ids)} != latents len {len(Z)}"

    # Load model
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    hidden = ckpt["config"]["HIDDEN"]
    dropout = ckpt["config"].get("DROPOUT", 0.0)

    trunk = Trunk(Z.shape[1], hidden, dropout)
    heads = Heads(hidden)
    trunk.load_state_dict(ckpt["trunk"])
    heads.load_state_dict(ckpt["heads"])
    trunk.eval(); heads.eval()

    with torch.no_grad():
        X = torch.from_numpy(Z)
        H = trunk(X)
        logits = heads(H)

    # Convert to unified labels per dataset
    start_map = {"Covertype": 0, "Higgs": 7, "HELOC": 9}
    y_pred_unified = np.empty(len(Z), dtype=int)
    for ds_idx, name in enumerate(order):
        m = (ds_ids == ds_idx)
        if not np.any(m):
            continue
        local = logits[name][m].argmax(dim=1).cpu().numpy().astype(int)
        y_pred_unified[m] = local + start_map[name]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "dataset_id": ds_ids.astype(int),
        "y_pred_unified": y_pred_unified.astype(int),
    }).to_csv(OUT_CSV, index=False)

    # Log counts
    uniq, cnt = np.unique(ds_ids, return_counts=True)
    print(f"dataset_id distribution: {dict(zip(uniq, cnt))}")
    print(f"Saved predictions -> {OUT_CSV}")

if __name__ == "__main__":
    main()
