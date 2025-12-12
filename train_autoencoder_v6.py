# -*- coding: utf-8 -*-
# train_autoencoder_v6.py
#
# Trains a shared autoencoder on the v6 unified feature space.
# AE input: [numeric_z | binary | native_mask | struct_mask]  (no dataset one-hot).
# Balanced training: each dataset yields the same number of mini-batches per epoch
# via RandomSampler(replacement=True, num_samples=target_samples).
#
# Saves:
#   models/encoder.pt            (encoder weights only)
#   models/autoencoder.pt        (full AE for reconstruction if needed)
#   merged_data/trainval_latents_v6.npy  (optional: latents for the full v6 matrix)

from pathlib import Path
import os, math, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

# ----------------- config -----------------
SEED = 42
LATENT_DIM = 64
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 50
PATIENCE = 8

# replace the current MERGED_DIR / MODELS_DIR block with this:
from pathlib import Path

try:
    from tabster_paths import MERGED_DIR, MODELS_DIR
except Exception:
    THIS_DIR = Path(__file__).resolve().parent
    MERGED_DIR = THIS_DIR / "merged_data"
    MODELS_DIR = THIS_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
NPZ_TRAIN = MERGED_DIR / "unified_data_v6.npz"

# ----------------- utils -----------------
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AEDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): 
        x = self.X[i]
        return x, x  # autoencoder: target == input

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128),       nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512),        nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def _split_blocks_v6(X, npz):
    """Slice X into v6 blocks using the lengths saved in the npz."""
    n_num  = len(npz["numeric_features"])
    n_bin  = len(npz["binary_features"])
    n_ds   = len(npz["dataset_onehot_cols"])
    n_nat  = len(npz["mask_native_cols"])
    n_str  = len(npz["mask_struct_cols"])
    i = 0
    X_num = X[:, i:i+n_num]; i += n_num
    X_bin = X[:, i:i+n_bin]; i += n_bin
    X_ds1 = X[:, i:i+n_ds];  i += n_ds
    X_nat = X[:, i:i+n_nat]; i += n_nat
    X_str = X[:, i:i+n_str]; i += n_str
    assert i == X.shape[1], "Block slicing mismatch with v6 layout"
    return X_num, X_bin, X_ds1, X_nat, X_str

@torch.no_grad()
def reconstruction_mse(model, X, device):
    loss_fn = nn.MSELoss(reduction="mean")
    mean_loss = 0.0
    bs = 4096
    n = len(X)
    for i in range(0, n, bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
        out, _ = model(xb)
        mean_loss += loss_fn(out, xb).item() * len(xb)
    return mean_loss / n

def _make_equalized_loader(X_block, batch_size, target_samples, device):
    """
    Build a DataLoader that yields a fixed number of samples per epoch
    regardless of dataset size by sampling with replacement.
    """
    ds = AEDataset(X_block)
    sampler = RandomSampler(ds, replacement=True, num_samples=int(target_samples))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)

# ----------------- main -----------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load v6 unified arrays
    npz = np.load(NPZ_TRAIN, allow_pickle=True)
    X = npz["X"]                    # [numeric_z | binary | ds_onehot | native | struct]
    ds_ids = npz["dataset_ids"]     # 0=covtype, 1=higgs, 2=heloc
    order  = [str(x) for x in npz["order"]]

    # Build AE input = numeric_z | binary | native_mask | struct_mask  (drop dataset one-hot)
    X_num, X_bin, _, X_nat, X_str = _split_blocks_v6(X, npz)
    X_ae = np.concatenate([X_num, X_bin, X_nat, X_str], axis=1).astype(np.float32)
    input_dim = X_ae.shape[1]

    # Per-dataset train/val split to keep representation stable
    rng = np.random.default_rng(SEED)
    Xtr_parts, Xva_parts = [], []
    id_tr_parts, id_va_parts = [], []
    for k, name in enumerate(order):
        idx_k = np.where(ds_ids == k)[0]
        idx_k = rng.permutation(idx_k)
        cut = int(0.9 * len(idx_k))
        tr_idx, va_idx = idx_k[:cut], idx_k[cut:]
        Xtr_parts.append(X_ae[tr_idx])
        Xva_parts.append(X_ae[va_idx])
        id_tr_parts.append(np.full(len(tr_idx), k, dtype=int))
        id_va_parts.append(np.full(len(va_idx), k, dtype=int))

    # Equalize per-epoch exposure across datasets via replacement sampling
    n_tr = [len(p) for p in Xtr_parts]
    target_samples = max(n_tr)  # each dataset contributes equally many samples per epoch
    train_loaders = [
        _make_equalized_loader(Xtr_parts[i], BATCH_SIZE, target_samples, device)
        for i in range(len(order))
    ]

    # Validation: compute per-dataset mean MSE and then unweighted average
    val_sets = [(Xva_parts[i], order[i]) for i in range(len(order))]

    # Model, opt, loss
    ae = AutoEncoder(input_dim=input_dim, latent_dim=LATENT_DIM).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad = 0

    # Steps per epoch equals loader length (all equal by construction)
    steps_per_epoch = len(train_loaders[0])

    for epoch in range(1, EPOCHS + 1):
        ae.train()
        epoch_loss = 0.0
        iters = [iter(dl) for dl in train_loaders]
        for _ in range(steps_per_epoch):
            # one mini-batch from each dataset, then average their losses
            losses = 0.0
            for it in iters:
                xb, yb = next(it)
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                out, _ = ae(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                losses += loss.item()
            epoch_loss += losses / len(iters)
        epoch_loss /= steps_per_epoch

        # Validation: unweighted mean across datasets to avoid size dominance
        ae.eval()
        with torch.no_grad():
            val_losses = []
            for Xv, name in val_sets:
                v = reconstruction_mse(ae, Xv, device)
                val_losses.append(v)
            val_mean = float(np.mean(val_losses))

        print(f"Epoch {epoch:03d} | train_mse={epoch_loss:.6f} | val_mse(mean across ds)={val_mean:.6f}")

        if val_mean + 1e-9 < best_val:
            best_val = val_mean
            best_state = copy.deepcopy(ae.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            print("Early stopping.")
            break

    # Restore best weights
    ae.load_state_dict(best_state)
    print(f"Best validation MSE (mean across datasets): {best_val:.6f}")

    # Save encoder + full AE
    torch.save({k.replace("encoder.", "", 1): v for k, v in ae.state_dict().items()
                if k.startswith("encoder.")}, MODELS_DIR / "encoder.pt")
    torch.save(ae.state_dict(), MODELS_DIR / "autoencoder.pt")
    print("Saved models/encoder.pt and models/autoencoder.pt")

    # Report full-dataset per-domain reconstruction MSE
    with torch.no_grad():
        for k, name in enumerate(order):
            Xk = X_ae[ds_ids == k]
            mse_k = reconstruction_mse(ae, Xk, device)
            print(f"{name} reconstruction MSE: {mse_k:.6f}")

    # Optional: export latents for the entire unified matrix (handy for downstream training)
    bs = 4096
    Z_all = []
    ae.eval()
    with torch.no_grad():
        for i in range(0, len(X_ae), bs):
            xb = torch.tensor(X_ae[i:i+bs], dtype=torch.float32, device=device)
            _, z = ae(xb)
            Z_all.append(z.cpu().numpy())
    Z_all = np.vstack(Z_all)
    np.save(MERGED_DIR / "trainval_latents_v6.npy", Z_all)
    print(f"Saved latents → {MERGED_DIR / 'trainval_latents_v6.npy'} with shape {Z_all.shape}")

if __name__ == "__main__":
    main()
