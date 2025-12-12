# scripts/build_and_encode_test_v6.py
# Build unified TEST features using training scaler/imputer and encode to latents.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Make sure we can import tabster_paths and dataset_merge_unified_v6
# ---------------------------------------------------------------------
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from tabster_paths import (
    DATA_DIR,
    MERGED_DIR,
    UNIFIED_NPZ,
    ZSCALE_PARAMS,
    IMPUTER_BINARY,
    TEST_PREPROCESSED,
    TEST_LATENTS,
    MODELS_DIR,
)

# Import helpers and constants from the merge script
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "dataset_merge_unified_v6.py").exists():
        sys.path.insert(0, str(cand))
        break

from dataset_merge_unified_v6 import load_and_clean, dataset_specific_fe, ORDER

# Encoder checkpoint path (this is a *checkpoint dict*, not a Module)
ENCODER_PATH = MODELS_DIR / "encoder.pt"

# Kaggle TEST CSVs
TEST_PATHS = {
    "Covertype": DATA_DIR / "covtype_test.csv",
    "Higgs":     DATA_DIR / "higgs_test.csv",
    "HELOC":     DATA_DIR / "heloc_test.csv",
}


def p(msg: str) -> None:
    print(msg, flush=True)


def build_unified_test():
    """
    Use TRAIN artifacts (UNIFIED_NPZ, ZSCALE_PARAMS, IMPUTER_BINARY)
    to build the unified test design matrix X_test and dataset_ids.
    """
    assert UNIFIED_NPZ.exists(), f"Missing training NPZ: {UNIFIED_NPZ}"
    assert ZSCALE_PARAMS.exists(), f"Missing numeric z-scaler params: {ZSCALE_PARAMS}"
    assert IMPUTER_BINARY.exists(), f"Missing binary imputer: {IMPUTER_BINARY}"

    npz = np.load(UNIFIED_NPZ, allow_pickle=True)
    features = npz["features"].astype(str).tolist()
    numeric_cols = npz["numeric_features"].astype(str).tolist()
    binary_cols = npz["binary_features"].astype(str).tolist()
    order = npz["order"].astype(str).tolist()

    p(f"Loaded training meta from {UNIFIED_NPZ}")
    p(f"- #features: {len(features)}")
    p(f"- #numeric : {len(numeric_cols)}")
    p(f"- #binary  : {len(binary_cols)}")
    p(f"- ORDER    : {order}")

    if list(order) != list(ORDER):
        p(f"[WARN] ORDER mismatch between NPZ {order} and code {ORDER}. Using NPZ order.")

    X_blocks = []
    native_blocks = []
    struct_blocks = []
    ds_ids_blocks = []

    for ds_idx, name in enumerate(order):
        if name not in TEST_PATHS:
            raise FileNotFoundError(f"No TEST_PATHS entry for dataset '{name}'")
        csv_path = TEST_PATHS[name]
        assert csv_path.exists(), f"Missing test CSV for {name}: {csv_path}"

        p(f"Loading test {name} from {csv_path}")
        df0, _ = load_and_clean(name, csv_path)
        df_fe = dataset_specific_fe(name, df0)

        # Align to training features
        # (yes, this is column-by-column and a bit slow, but it's safe)
        dfa = pd.DataFrame(index=df_fe.index)
        present = set(df_fe.columns)
        for f in features:
            if f in df_fe.columns:
                dfa[f] = df_fe[f]
            else:
                dfa[f] = np.nan

        # struct mask: 0 if feature exists in this dataset, 1 if absent
        struct = pd.DataFrame(
            {f: (0.0 if f in present else 1.0) for f in features},
            index=dfa.index,
            dtype=np.float32,
        )
        # native mask: NaN in existing feature (struct==0)
        native = dfa[features].isna().astype(np.float32) * (1.0 - struct.values)

        X_blocks.append(dfa[features])
        native_blocks.append(native)
        struct_blocks.append(struct)
        ds_ids_blocks.append(np.full(len(dfa), ds_idx, dtype=int))

        p(f"- {name} test rows: {len(dfa)}")

    X_concat = pd.concat(X_blocks, axis=0)
    native_concat = pd.concat(native_blocks, axis=0).to_numpy(dtype=np.float32)
    struct_concat = pd.concat(struct_blocks, axis=0).to_numpy(dtype=np.float32)
    ds_ids = np.concatenate(ds_ids_blocks)

    p(f"Unified TEST raw shape: {X_concat.shape}")
    p(f"Native mask shape      : {native_concat.shape}")
    p(f"Struct mask shape      : {struct_concat.shape}")

    # Numeric scaling
    zs = np.load(ZSCALE_PARAMS)
    mu, sigma = zs["mu"], zs["sigma"]
    numeric_feats_z = zs["numeric_features"].astype(str).tolist()
    if numeric_feats_z != numeric_cols:
        p("[WARN] numeric_features in ZSCALE_PARAMS differ from UNIFIED_NPZ; using ZSCALE_PARAMS order.")
        numeric_cols = numeric_feats_z  # follow scaler order

    if numeric_cols:
        X_num_raw = X_concat[numeric_cols].to_numpy().astype(np.float32)
        sigma_safe = np.where(sigma == 0.0, 1.0, sigma)
        X_num_z = (X_num_raw - mu) / sigma_safe
        X_num = np.nan_to_num(X_num_z, nan=0.0).astype(np.float32)
    else:
        X_num = np.empty((len(X_concat), 0), dtype=np.float32)

    # Binary imputation
    if binary_cols:
        imputer = joblib.load(IMPUTER_BINARY)
        X_bin = imputer.transform(X_concat[binary_cols]).astype(np.float32)
    else:
        X_bin = np.empty((len(X_concat), 0), dtype=np.float32)

    # Dataset one-hot
    n_ds = len(order)
    ds_onehot = np.eye(n_ds, dtype=np.float32)[ds_ids]

    # Final design matrix
    X_final = np.concatenate(
        [X_num, X_bin, ds_onehot, native_concat, struct_concat],
        axis=1,
    )

    p(f"Final TEST X shape: {X_final.shape}")

    # Persist preprocessed test (for debugging & later reuse)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        TEST_PREPROCESSED,
        X=X_final.astype(np.float32),
        dataset_ids=ds_ids.astype(int),
        order=np.array(order),
        features=np.array(features),
        numeric_features=np.array(numeric_cols),
        binary_features=np.array(binary_cols),
    )
    p(f"Saved preprocessed test → {TEST_PREPROCESSED}")

    return X_final.astype(np.float32), ds_ids.astype(int)


# ---------------------------------------------------------------------
# Rebuild encoder module from a checkpoint dict
# ---------------------------------------------------------------------
def _build_encoder_from_checkpoint(ckpt: dict, device: torch.device) -> nn.Module:
    """
    ckpt is a dict (what you currently have in encoder.pt).
    We try to extract encoder weights and rebuild a plain feedforward encoder.

    Supported patterns:
      - ckpt['encoder'] is a state_dict (name -> tensor)
      - ckpt['state_dict'] has keys starting with 'encoder.'
    """
    # 1) Get encoder state_dict
    if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
        enc_sd = ckpt["encoder"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        full_sd = ckpt["state_dict"]
        enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
        if not enc_sd:
            raise RuntimeError("Found state_dict but no keys starting with 'encoder.'")
    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        # Probably a bare state_dict for the encoder (keys like '0.weight', '0.bias', ...)
        enc_sd = ckpt
    else:
        raise RuntimeError(
            f"Cannot find encoder weights in checkpoint. Keys are: {list(ckpt.keys())}"
        )

    # 2) Extract weight/bias tensors, sorted by name to keep layer order
    weight_items = [(k, v) for k, v in enc_sd.items() if k.endswith(".weight")]
    bias_items = [(k, v) for k, v in enc_sd.items() if k.endswith(".bias")]

    if not weight_items or not bias_items or len(weight_items) != len(bias_items):
        raise RuntimeError(
            "Encoder state_dict doesn't have matching weight/bias pairs. "
            f"weights: {len(weight_items)}, biases: {len(bias_items)}"
        )

    weight_items.sort(key=lambda kv: kv[0])
    bias_items.sort(key=lambda kv: kv[0])

    layers: list[nn.Module] = []
    for i, ((w_name, w), (b_name, b)) in enumerate(zip(weight_items, bias_items)):
        out_features, in_features = w.shape
        if b.shape[0] != out_features:
            raise RuntimeError(
                f"Shape mismatch between {w_name} and {b_name}: "
                f"weight {w.shape}, bias {b.shape}"
            )

        lin = nn.Linear(in_features, out_features)
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        # Add ReLU between all linear layers except after the last one
        if i < len(weight_items) - 1:
            layers.append(nn.ReLU())

    encoder = nn.Sequential(*layers).to(device)
    return encoder


def encode_to_latents(X_test: np.ndarray):
    """
    Load the trained encoder checkpoint (a dict), rebuild an encoder Module,
    and map X_test to latent space.
    """
    assert ENCODER_PATH.exists(), f"Missing encoder checkpoint: {ENCODER_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"Using device: {device}")

    ckpt = torch.load(ENCODER_PATH, map_location=device)

    if isinstance(ckpt, nn.Module):
        # In case you ever save the bare module instead of a dict in the future
        encoder = ckpt.to(device)
    elif isinstance(ckpt, dict):
        encoder = _build_encoder_from_checkpoint(ckpt, device)
    else:
        raise TypeError(
            f"Unexpected type in ENCODER_PATH: {type(ckpt)}. "
            "Expected nn.Module or dict with encoder weights."
        )

    encoder.eval()

    # The autoencoder was trained on [numeric_z | binary | native_mask | struct_mask]
    # (dataset one-hot was NOT included). Slice X_test accordingly using training meta.
    npz_meta = np.load(UNIFIED_NPZ, allow_pickle=True)
    n_num = len(npz_meta["numeric_features"])
    n_bin = len(npz_meta["binary_features"])
    n_ds  = len(npz_meta["dataset_onehot_cols"])
    n_nat = len(npz_meta["mask_native_cols"])
    n_str = len(npz_meta["mask_struct_cols"])
    i = 0
    X_num = X_test[:, i:i+n_num]; i += n_num
    X_bin = X_test[:, i:i+n_bin]; i += n_bin
    _ds1  = X_test[:, i:i+n_ds];  i += n_ds
    X_nat = X_test[:, i:i+n_nat]; i += n_nat
    X_str = X_test[:, i:i+n_str]; i += n_str
    X_ae = np.concatenate([X_num, X_bin, X_nat, X_str], axis=1).astype(np.float32, copy=False)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_ae).to(device)
        Z = encoder(X_tensor).cpu().numpy().astype(np.float32)

    np.save(TEST_LATENTS, Z)
    p(f"Saved TEST latents → {TEST_LATENTS}")
    p(f"Latent shape: {Z.shape}")

    return Z


def main():
    print("=" * 80)
    print("BUILD + ENCODE KAGGLE TEST (v6)")
    print("=" * 80)

    X_test, ds_ids = build_unified_test()
    encode_to_latents(X_test)


if __name__ == "__main__":
    main()
