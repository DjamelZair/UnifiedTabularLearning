# evaluate_autoencoder_v6.py
# One-run pipeline that reuses your existing modules.
# - Reuse training: import train_autoencoder_v6 and call main() if AE missing
# - Reuse preprocessing: import preprocess_unified_v6 and call main() if TEST npz missing
# - Evaluate reconstruction on TEST and save latents

from __future__ import annotations
import sys
from pathlib import Path
import importlib
import numpy as np
import torch
import torch.nn as nn

# ---------- paths via your tabster_paths ----------
try:
    from tabster_paths import PROJECT_ROOT, MERGED_DIR, DATA_DIR, MODELS_DIR
except Exception:
    THIS_DIR = Path(__file__).resolve().parent
    # simple fallback if tabster_paths not found
    PROJECT_ROOT = THIS_DIR
    MERGED_DIR   = PROJECT_ROOT / "merged_data"
    DATA_DIR     = PROJECT_ROOT / "datasets"
    MODELS_DIR   = PROJECT_ROOT / "models"

AE_PATH          = MODELS_DIR / "autoencoder.pt"
TRAIN_UNIFIED_NPZ = MERGED_DIR / "unified_data_v6.npz"
TEST_NPZ_PATH     = MERGED_DIR / "test_preprocessed_v6.npz"
TEST_LATENTS_NPY  = MERGED_DIR / "test_latents_v6.npy"

# ---------- import your training module and reuse its class ----------
try:
    train_mod = importlib.import_module("train_autoencoder_v6")
except ModuleNotFoundError as e:
    raise RuntimeError("Cannot import train_autoencoder_v6.py. Place evaluate_autoencoder_v6.py in the same folder or add it to PYTHONPATH.") from e

AutoEncoder = train_mod.AutoEncoder
LATENT_DIM  = getattr(train_mod, "LATENT_DIM", 64)

def ensure_trained_autoencoder():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if AE_PATH.exists():
        print(f"AE already present at {AE_PATH}")
        return
    # call your trainer's main() to produce models/autoencoder.pt
    print("No AE found. Calling train_autoencoder_v6.main() …", flush=True)
    train_mod.main()
    if not AE_PATH.exists():
        # trainer saved to its own MODELS_DIR; still missing here means inconsistent roots
        raise FileNotFoundError(f"Expected AE at {AE_PATH} after training. Check tabster_paths detection and working directory.")

def ensure_test_npz():
    if TEST_NPZ_PATH.exists():
        return
    print("No TEST AE matrix. Calling preprocess_unified_v6.main() …", flush=True)
    try:
        pre_mod = importlib.import_module("preprocess_unified_v6")
    except ModuleNotFoundError as e:
        raise RuntimeError("Cannot import preprocess_unified_v6.py. Put it next to this file or on PYTHONPATH.") from e
    # call its main() with defaults by simulating an empty argv
    old_argv = sys.argv[:]
    try:
        sys.argv = [pre_mod.__file__]
        pre_mod.main()
    finally:
        sys.argv = old_argv
    if not TEST_NPZ_PATH.exists():
        raise FileNotFoundError(f"Expected TEST AE npz at {TEST_NPZ_PATH} after preprocessing.")

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
    out = []
    for name, (a, b) in spans.items():
        if b > a:
            mse = float(np.mean((X_true[:, a:b] - X_pred[:, a:b])**2))
            out.append((name, f"{mse:.6f}"))
    return out

def evaluate_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_npz = np.load(TEST_NPZ_PATH, allow_pickle=True)

    X_ae  = test_npz["X_ae"].astype(np.float32)
    ds_ids = test_npz["dataset_ids"].astype(int)
    order  = [str(x) for x in test_npz["order"]]

    n_num = len(test_npz["numeric_features"])
    n_bin = len(test_npz["binary_features"])
    n_nat = len(test_npz["mask_native_cols"])
    n_str = len(test_npz["mask_struct_cols"])
    D_ae  = n_num + n_bin + n_nat + n_str
    if D_ae != X_ae.shape[1]:
        raise RuntimeError(f"AE input dim mismatch: expected {D_ae}, got {X_ae.shape[1]}")

    # load AE using your imported class
    ae = AutoEncoder(input_dim=D_ae, latent_dim=LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae.eval()

    Xhat, Z = _batched_forward_recon(ae, X_ae, device=device, bs=4096)

    overall_mse  = float(np.mean((X_ae - Xhat)**2))
    overall_rmse = float(np.sqrt(np.mean((X_ae - Xhat)**2)))

    # per-dataset MSE
    per_ds = []
    for k, name in enumerate(order):
        m = (ds_ids == k)
        if m.any():
            mval = float(np.mean((X_ae[m] - Xhat[m])**2))
            per_ds.append((name, f"{mval:.6f}", int(m.sum())))
    per_ds.sort(key=lambda x: float(x[1]), reverse=True)

    # block spans and per-block MSE
    spans = {
        "numeric_z":  (0, n_num),
        "binary":     (n_num, n_num + n_bin),
        "native_msk": (n_num + n_bin, n_num + n_bin + n_nat),
        "struct_msk": (n_num + n_bin + n_nat, n_num + n_bin + n_nat + n_str),
    }
    block_rows = _summarize_blocks(X_ae, Xhat, spans)

    # baseline mean from TRAIN AE space
    train_npz = np.load(TRAIN_UNIFIED_NPZ, allow_pickle=True)
    i = 0
    n_num_tr = len(train_npz["numeric_features"])
    n_bin_tr = len(train_npz["binary_features"])
    n_ds_tr  = len(train_npz["dataset_onehot_cols"])
    n_nat_tr = len(train_npz["mask_native_cols"])
    n_str_tr = len(train_npz["mask_struct_cols"])
    Xtr_full = train_npz["X"]
    Xnum_tr = Xtr_full[:, i:i+n_num_tr]; i += n_num_tr
    Xbin_tr = Xtr_full[:, i:i+n_bin_tr]; i += n_bin_tr
    _      = Xtr_full[:, i:i+n_ds_tr];  i += n_ds_tr
    Xnat_tr = Xtr_full[:, i:i+n_nat_tr]; i += n_nat_tr
    Xstr_tr = Xtr_full[:, i:i+n_str_tr]; i += n_str_tr
    mu_train = np.mean(np.concatenate([Xnum_tr, Xbin_tr, Xnat_tr, Xstr_tr], axis=1), axis=0).astype(np.float32)

    X_base = np.broadcast_to(mu_train, X_ae.shape)
    base_mse = float(np.mean((X_ae - X_base)**2))
    base_blocks = _summarize_blocks(X_ae, X_base, spans)

    # latent variance
    var_latent = Z.var(axis=0)
    latent_summary = {
        "mean(var)": float(var_latent.mean()),
        "min(var)":  float(var_latent.min()),
        "max(var)":  float(var_latent.max()),
        "num_near_zero(var<1e-5)": int((var_latent < 1e-5).sum()),
    }

    # pretty tables
    def print_table(rows, header):
        if not rows:
            print("(no rows)"); return
        w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt = " | ".join("{:" + str(width) + "}" for width in w)
        sep = "-+-".join("-" * width for width in w)
        print(fmt.format(*header)); print(sep)
        for r in rows: print(fmt.format(*r))

    print("\n=== Autoencoder Reconstruction on TEST (v6) ===")
    print(f"TEST rows: {len(X_ae):,} | AE input dim: {D_ae}")
    print(f"Overall    MSE:  {overall_mse:.6f}")
    print(f"Overall   RMSE:  {overall_rmse:.6f}")
    print(f"Baseline   MSE:  {base_mse:.6f}  (constant mean from TRAIN v6)")
    print(f"AE vs Baseline ΔMSE: {base_mse - overall_mse:.6f}  (positive = AE better)")

    print("\n-- Per-dataset reconstruction MSE (higher = worse) --")
    print_table(per_ds, header=["Dataset","MSE","N"])

    print("\n-- Per-block MSE (AE) --")
    print_table(block_rows, header=["Block","MSE"])

    print("\n-- Per-block MSE (Baseline) --")
    print_table(base_blocks, header=["Block","MSE"])

    print("\n-- Latent variance summary --")
    for k, v in latent_summary.items():
        print(f"{k}: {v}")

    np.save(TEST_LATENTS_NPY, Z.astype(np.float32, copy=False))
    print(f"\nSaved TEST latents -> {TEST_LATENTS_NPY} with shape {Z.shape}")

def main():
    ensure_trained_autoencoder()
    ensure_test_npz()
    evaluate_and_save()

if __name__ == "__main__":
    main()
