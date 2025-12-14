# -*- coding: utf-8 -*-
# tsne_latents_v6.py
#
# Generates embeddings using models/encoder.pt and visualizes them with t-SNE.
# Colors points by dataset_id (covtype/higgs/heloc) from unified_data_v6.npz.

from pathlib import Path
import inspect
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ----------------- config -----------------
LATENT_DIM   = 64
BATCH_SIZE   = 4096

# t-SNE params (tweak if needed)
TSNE_PERPLEXITY = 30
TSNE_LR         = "auto"   # sklearn supports "auto" in newer versions
TSNE_N_ITER     = 1000
TSNE_SEED       = 42

# Color palette for the three datasets (covtype, higgs, heloc by default).
# Edit these hex values to try your own palette; extra datasets will cycle.
COLOR_PALETTE = [
    "#267d21ff",  # covtype
    "#d5cb4b",  # higgs
    "#4f99cf",  # heloc
    # "covtype":   "#267d21ff",
    # "higgs":     "#d5cb4b",
    # "heloc":     "#4f99cf",
]

# For speed: optionally subsample points before t-SNE
MAX_POINTS = None # 20000  # set None to use all points (can be very slow)

# ----------------- paths -----------------
try:
    from tabster_paths import MERGED_DIR, MODELS_DIR
except Exception:
    THIS_DIR = Path(__file__).resolve().parent
    MERGED_DIR = THIS_DIR / "merged_data"
    MODELS_DIR = THIS_DIR / "models"

NPZ_PATH     = MERGED_DIR / "unified_data_v6.npz"
ENCODER_PATH = MODELS_DIR / "encoder.pt"

# ----------------- model -----------------
class Encoder(nn.Module):
    """Must match the encoder architecture used during training."""
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128),       nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

def split_blocks_v6(X, npz):
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
def encode_all(encoder, X_ae, device, batch_size=BATCH_SIZE):
    Z = []
    for i in range(0, len(X_ae), batch_size):
        xb = torch.tensor(X_ae[i:i+batch_size], dtype=torch.float32, device=device)
        zb = encoder(xb)
        Z.append(zb.cpu().numpy())
    return np.vstack(Z)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load npz ----
    npz = np.load(NPZ_PATH, allow_pickle=True)
    X      = npz["X"]
    ds_ids = npz["dataset_ids"].astype(int)   # 0,1,2
    order  = [str(x) for x in npz["order"]]   # names in order

    # ---- build AE input (same as training script) ----
    X_num, X_bin, _, X_nat, X_str = split_blocks_v6(X, npz)
    X_ae = np.concatenate([X_num, X_bin, X_nat, X_str], axis=1).astype(np.float32)
    input_dim = X_ae.shape[1]
    print("X_ae shape:", X_ae.shape)

    # ---- load encoder ----
    encoder = Encoder(input_dim=input_dim, latent_dim=LATENT_DIM).to(device)

    # Handle older checkpoints that saved encoder weights without the "net." prefix.
    state = torch.load(ENCODER_PATH, map_location=device)
    needs_prefix = all(not k.startswith("net.") for k in state.keys())
    if needs_prefix:
        state = {f"net.{k}": v for k, v in state.items()}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    # ---- encode ----
    Z_all = encode_all(encoder, X_ae, device)
    latents_path = MERGED_DIR / "latents_v6.npy"
    np.save(latents_path, Z_all)
    print(f"Saved latents → {latents_path} {Z_all.shape}")

    # ---- subsample for t-SNE (recommended) ----
    n = len(Z_all)
    if MAX_POINTS is not None and n > MAX_POINTS:
        rng = np.random.default_rng(TSNE_SEED)
        idx = rng.choice(n, size=MAX_POINTS, replace=False)
        Z = Z_all[idx]
        y = ds_ids[idx]
        print(f"Subsampled {MAX_POINTS}/{n} points for t-SNE")
    else:
        Z = Z_all
        y = ds_ids

    # ---- t-SNE ----
    tsne_kwargs = dict(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, max(5, (len(Z) - 1) // 3)),
        learning_rate=TSNE_LR,
        init="pca",
        random_state=TSNE_SEED,
    )
    # scikit-learn renamed n_iter → max_iter in newer versions; handle both.
    tsne_sig = inspect.signature(TSNE.__init__)
    if "n_iter" in tsne_sig.parameters:
        tsne_kwargs["n_iter"] = TSNE_N_ITER
    elif "max_iter" in tsne_sig.parameters:
        tsne_kwargs["max_iter"] = TSNE_N_ITER
    tsne = TSNE(**tsne_kwargs)
    emb2 = tsne.fit_transform(Z)
    tsne_path = MERGED_DIR / "tsne_2d_v6.npy"
    np.save(tsne_path, emb2)
    print(f"Saved t-SNE coords → {tsne_path} {emb2.shape}")

    # ---- plot ----
    plt.figure(figsize=(10, 7))
    for k, name in enumerate(order):
        m = (y == k)
        color = COLOR_PALETTE[k % len(COLOR_PALETTE)]
        plt.scatter(emb2[m, 0], emb2[m, 1], s=6, alpha=0.6, label=name, color=color)

    plt.title("t-SNE of Shared Latent Representations Across Tabular Datasets")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=7, fontsize=18)
    plt.tight_layout()

    out_png = MERGED_DIR / "tsne_latents_v6.png"
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot → {out_png}")

    plt.show()

if __name__ == "__main__":
    main()
