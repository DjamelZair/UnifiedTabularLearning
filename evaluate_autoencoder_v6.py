# evaluate_autoencoder_v6.py
# Evaluate a trained AE on a stratified TEST hold-out built from unified_data_v6.npz.
# No dependency on any "preprocess_unified_v6". Uses the same block logic as the trainer.

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# ---------- paths via tabster_paths or local fallback ----------
try:
    from tabster_paths import PROJECT_ROOT, MERGED_DIR, MODELS_DIR
except Exception:
    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR
    MERGED_DIR   = PROJECT_ROOT / "merged_data"
    MODELS_DIR   = PROJECT_ROOT / "models"

AE_PATH                    = MODELS_DIR / "autoencoder.pt"
TRAIN_UNIFIED_NPZ          = MERGED_DIR / "unified_data_v6.npz"
# Keep internal AE hold-out latents separate from Kaggle test latents
INTERNAL_TEST_LATENTS_NPY  = MERGED_DIR / "internal_test_latents_v6.npy"

# ---------- import trainer types without side effects ----------
try:
    import train_autoencoder_v6 as trainer
except ModuleNotFoundError as e:
    raise RuntimeError("Place evaluate_autoencoder_v6.py next to train_autoencoder_v6.py or add it to PYTHONPATH.") from e

AutoEncoder = trainer.AutoEncoder
LATENT_DIM  = getattr(trainer, "LATENT_DIM", 64)

# ---------- helpers ----------
def ensure_trained_autoencoder():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if AE_PATH.exists():
        return
    print("No AE found. Training once via train_autoencoder_v6.main() …", flush=True)
    trainer.main()
    if not AE_PATH.exists():
        raise FileNotFoundError(f"Expected AE at {AE_PATH} after training. Check working directory and tabster_paths.")

def load_unified_v6():
    if not TRAIN_UNIFIED_NPZ.exists():
        raise FileNotFoundError(f"Missing {TRAIN_UNIFIED_NPZ}. Run dataset_merge_unified_v6.py first.")
    npz = np.load(TRAIN_UNIFIED_NPZ, allow_pickle=True)
    X_full   = npz["X"].astype(np.float32)
    ds_ids   = npz["dataset_ids"].astype(int)
    order    = [str(x) for x in npz["order"]]
    n_num    = len(npz["numeric_features"])
    n_bin    = len(npz["binary_features"])
    n_ds     = len(npz["dataset_onehot_cols"])
    n_nat    = len(npz["mask_native_cols"])
    n_str    = len(npz["mask_struct_cols"])
    # Build AE view: numeric_z | binary | native_mask | struct_mask  (drop dataset one-hot)
    i = 0
    X_num = X_full[:, i:i+n_num]; i += n_num
    X_bin = X_full[:, i:i+n_bin]; i += n_bin
    _     = X_full[:, i:i+n_ds];  i += n_ds
    X_nat = X_full[:, i:i+n_nat]; i += n_nat
    X_str = X_full[:, i:i+n_str]; i += n_str
    X_ae  = np.concatenate([X_num, X_bin, X_nat, X_str], axis=1).astype(np.float32)
    D_ae  = X_ae.shape[1]
    spans = {
        "numeric_z":  (0, n_num),
        "binary":     (n_num, n_num + n_bin),
        "native_msk": (n_num + n_bin, n_num + n_bin + n_nat),
        "struct_msk": (n_num + n_bin + n_nat, n_num + n_bin + n_nat + n_str),
    }
    return X_ae, ds_ids, order, D_ae, spans, npz

def stratified_holdout_by_dataset(ds_ids, ratio=0.10, seed=42):
    rng = np.random.default_rng(seed)
    idx_test, idx_rest = [], []
    for k in np.unique(ds_ids):
        rows = np.where(ds_ids == k)[0]
        rng.shuffle(rows)
        n_test = max(1, int(round(len(rows) * ratio)))
        idx_test.extend(rows[:n_test])
        idx_rest.extend(rows[n_test:])
    return np.array(idx_rest, dtype=int), np.array(idx_test, dtype=int)

@torch.no_grad()
def _batched_forward_recon(model, X, device, bs=4096):
    outs, zs = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
        out, z = model(xb)
        outs.append(out.cpu().numpy())
        zs.append(z.cpu().numpy())
    return np.vstack(outs), np.vstack(zs)

def _summarize_blocks(X_true, X_pred, spans):
    rows = []
    for name, (a, b) in spans.items():
        if b > a:
            mse = float(np.mean((X_true[:, a:b] - X_pred[:, a:b])**2))
            rows.append((name, f"{mse:.6f}"))
    return rows

def evaluate_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_ae_all, ds_ids_all, order, D_ae, spans, train_npz = load_unified_v6()
    # internal TEST hold-out by dataset only
    idx_rest, idx_test = stratified_holdout_by_dataset(ds_ids_all, ratio=0.10, seed=42)
    X_ae  = X_ae_all[idx_test]
    ds_ids = ds_ids_all[idx_test]

    ae = AutoEncoder(input_dim=D_ae, latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae.eval()

    Xhat, Z = _batched_forward_recon(ae, X_ae, device=device, bs=4096)

    overall_mse  = float(np.mean((X_ae - Xhat)**2))
    overall_rmse = float(np.sqrt(np.mean((X_ae - Xhat)**2)))

    per_ds = []
    for k, name in enumerate(order):
        m = (ds_ids == k)
        if m.any():
            mval = float(np.mean((X_ae[m] - Xhat[m])**2))
            per_ds.append((name, f"{mval:.6f}", int(m.sum())))
    per_ds.sort(key=lambda x: float(x[1]), reverse=True)

    block_rows = _summarize_blocks(X_ae, Xhat, spans)

    # constant-mean baseline built from TRAIN matrix in unified v6
    X_full_tr = train_npz["X"]
    n_num = len(train_npz["numeric_features"])
    n_bin = len(train_npz["binary_features"])
    n_ds  = len(train_npz["dataset_onehot_cols"])
    n_nat = len(train_npz["mask_native_cols"])
    n_str = len(train_npz["mask_struct_cols"])
    i = 0
    Xnum_tr = X_full_tr[:, i:i+n_num]; i += n_num
    Xbin_tr = X_full_tr[:, i:i+n_bin]; i += n_bin
    _      = X_full_tr[:, i:i+n_ds];  i += n_ds
    Xnat_tr = X_full_tr[:, i:i+n_nat]; i += n_nat
    Xstr_tr = X_full_tr[:, i:i+n_str]; i += n_str
    mu_train = np.mean(np.concatenate([Xnum_tr, Xbin_tr, Xnat_tr, Xstr_tr], axis=1), axis=0).astype(np.float32)
    X_base = np.broadcast_to(mu_train, X_ae.shape)
    base_mse = float(np.mean((X_ae - X_base)**2))
    base_blocks = _summarize_blocks(X_ae, X_base, spans)

    var_latent = Z.var(axis=0)
    latent_summary = {
        "mean(var)": float(var_latent.mean()),
        "min(var)":  float(var_latent.min()),
        "max(var)":  float(var_latent.max()),
        "num_near_zero(var<1e-5)": int((var_latent < 1e-5).sum()),
    }

    def print_table(rows, header):
        if not rows:
            print("(no rows)"); return
        w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt = " | ".join("{:" + str(width) + "}" for width in w)
        sep = "-+-".join("-" * width for width in w)
        print(fmt.format(*header)); print(sep)
        for r in rows:
            print(fmt.format(*r))

    print("\n=== Autoencoder Reconstruction on internal TEST (v6) ===")
    print(f"TEST rows: {len(X_ae):,} | AE input dim: {D_ae}")
    print(f"Overall    MSE:  {overall_mse:.6f}")
    print(f"Overall   RMSE:  {overall_rmse:.6f}")
    print(f"Baseline   MSE:  {base_mse:.6f}  (constant mean from unified TRAIN)")
    print(f"AE vs Baseline ΔMSE: {base_mse - overall_mse:.6f}  (positive means AE better)")

    print("\n-- Per-dataset reconstruction MSE (higher means worse) --")
    print_table(per_ds, header=["Dataset","MSE","N"])

    print("\n-- Per-block MSE (AE) --")
    print_table(block_rows, header=["Block","MSE"])

    print("\n-- Per-block MSE (Baseline) --")
    print_table(base_blocks, header=["Block","MSE"])

    print("\n-- Latent variance summary --")
    for k, v in latent_summary.items():
        print(f"{k}: {v}")

    np.save(INTERNAL_TEST_LATENTS_NPY, Z.astype(np.float32, copy=False))
    print(f"\nSaved INTERNAL TEST latents -> {INTERNAL_TEST_LATENTS_NPY} with shape {Z.shape}")

def main():
    ensure_trained_autoencoder()
    evaluate_and_save()

if __name__ == "__main__":
    main()
