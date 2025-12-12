# ==========================================================
# dataset_merge_unified_v6.py  (GLOBAL SCALING, KEEP HELOC ROWS)
# Unified table for Covertype, HIGGS, HELOC with:
# - Covertype FE: Aspect -> cos only; Elev_minus_VertHydro; no Hillshade_mean
# - HELOC: DO NOT DROP rows with NMAR codes (-7/-8/-9); convert codes -> NaN
# - Union align (adds structural NaNs where a feature doesn't exist)
# - GLOBAL numeric scaling: z-score using nan-robust stats over ALL rows per feature
#   then fill remaining NaNs with 0 (neutral in z; AE-safe)
# - Binary impute: most_frequent (configurable)
# - Dataset one-hots
# - TWO masks only:
#     1) <feat>_native_mask  (exists-in-dataset AND NaN after cleaning)
#     2) <feat>_struct_mask  (feature absent in dataset)
# - Persist feature ownership, HELOC NMAR stats, and z-scaler params
# ==========================================================

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# -----------------------------
# Project root & paths (robust to nesting)
# -----------------------------
def _detect_project_root() -> Path:
    here = Path(__file__).resolve().parent
    # Prefer the folder that contains this file if it has the expected subfolders
    if (here / "datasets").exists() and (here / "merged_data").exists():
        return here
    # Try one level up (useful if someone moved the script under a subfolder later)
    up1 = here.parent
    if (up1 / "datasets").exists() and (up1 / "merged_data").exists():
        return up1
    # Allow explicit override for CI/server runs
    env = os.getenv("TABSTER_PROJECT_ROOT")
    if env:
        cand = Path(env).resolve()
        if (cand / "datasets").exists() and (cand / "merged_data").exists():
            return cand
    # Fallback to this file's directory; user must ensure relative layout
    return here

PROJECT_ROOT = _detect_project_root()

# -----------------------------
# Configuration
# -----------------------------
DATA_PATHS = {
    "Covertype": PROJECT_ROOT / "datasets" / "covtype_train.csv",
    "Higgs":     PROJECT_ROOT / "datasets" / "higgs_train.csv",
    "HELOC":     PROJECT_ROOT / "datasets" / "heloc_train.csv",
}
TARGETS = {"Covertype": "Cover_Type", "Higgs": "Label", "HELOC": "RiskPerformance"}

# Simulator / vendor missing encodings
MISSING_ENCODING = {"Covertype": [], "Higgs": [-999], "HELOC": [-7, -8, -9]}
ORDER = ["Covertype", "Higgs", "HELOC"]

# Optional: light winsorization before scaling (per-dataset, numerics only)
WINSORIZE = False
WINSOR_LIMITS = (0.01, 0.99)

# Binary imputation policy
BINARY_IMPUTE_STRATEGY = "most_frequent"  # or "constant_zero"

OUTDIR = PROJECT_ROOT / "merged_data"
OUTDIR.mkdir(parents=True, exist_ok=True)

# v6 artifact paths
NPZ_PATH = OUTDIR / "unified_data_v6.npz"
ZSCALE_PARAMS_PATH = OUTDIR / "zscaler_numeric_params_v6.npz"
IMPUTER_BIN_PATH = OUTDIR / "imputer_binary_v6.pkl"
FEATURE_CATALOG_PATH = OUTDIR / "feature_catalog_v6.csv"
META_PATH = OUTDIR / "merge_meta_v6.json"

# -----------------------------
# Helpers
# -----------------------------
def coerce_higgs_label(s: pd.Series) -> pd.Series:
    """Map HIGGS labels to {'b','s'} irrespective of 0/1 or strings."""
    if s.dtype.kind in "iu":
        return s.map({0: "b", 1: "s"}).astype("category")
    if s.dtype.kind == "f":
        return s.round().astype(int).map({0: "b", 1: "s"}).astype("category")
    return (
        s.astype(str).str.strip().str.lower()
         .map({"0": "b", "1": "s", "b": "b", "s": "s"})
         .astype("category")
    )

def load_and_clean(name: str, path: Path) -> Tuple[pd.DataFrame, dict]:
    """Dataset-specific cleaning; returns df and info dict."""
    info = {}
    df = pd.read_csv(path)

    if name == "Higgs":
        # labels, drop non-features
        if TARGETS[name] in df.columns:
            df[TARGETS[name]] = coerce_higgs_label(df[TARGETS[name]])
        for c in ["EventId", "Weight", "KaggleSet", "KaggleWeight"]:
            if c in df.columns:
                df = df.drop(columns=c)
        # replace simulator sentinel with NaN (true missing)
        df = df.replace(-999, np.nan)

    elif name == "HELOC":
        # v6 policy: keep rows, convert NMAR codes to NaN, log stats
        codes = MISSING_ENCODING["HELOC"]
        n_before = len(df)
        code_counts = {str(c): int((df == c).sum().sum()) for c in codes}
        rows_with_any_code = int(df.isin(set(codes)).any(axis=1).sum())
        df = df.replace({-7: np.nan, -8: np.nan, -9: np.nan})
        info["heloc_rows_kept_with_codes"] = rows_with_any_code
        info["heloc_total_rows"] = int(n_before)
        info["heloc_code_counts"] = code_counts

    elif name == "Covertype":
        # no special missing encodings
        pass

    return df, info

def fe_covtype(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative feature engineering for Covertype per ablation."""
    dfe = df.copy()

    # Aspect -> cos only; drop raw Aspect
    if "Aspect" in dfe.columns:
        aspect = dfe["Aspect"].astype(float) % 360.0
        dfe["Aspect_cos"] = np.cos(np.deg2rad(aspect))
        dfe = dfe.drop(columns=["Aspect"])

    # Elev_minus_VertHydro
    if {"Elevation", "Vertical_Distance_To_Hydrology"}.issubset(dfe.columns):
        dfe["Elev_minus_VertHydro"] = (
            dfe["Elevation"].astype(float) - dfe["Vertical_Distance_To_Hydrology"].astype(float)
        )
    return dfe

def dataset_specific_fe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name == "Covertype":
        return fe_covtype(df)
    return df

def build_union_features(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    """Union of all feature names excluding targets."""
    cols = set()
    for n, d in dfs.items():
        cols |= set(d.columns) - {TARGETS[n]}
    return sorted(cols)

def align(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """Ensure every dataset has the union of features in the same order."""
    dfa = df.copy()
    for col in features:
        if col not in dfa.columns:
            dfa[col] = np.nan
    return dfa[features + [target]]

def detect_binaries(aligned: Dict[str, pd.DataFrame], features: List[str]) -> List[bool]:
    """Global binary test: a column is binary if all non-NaN values across datasets are in {0,1}."""
    flags = []
    for c in features:
        vals = []
        for _, d in aligned.items():
            u = d[c].dropna().unique()
            if len(u):
                vals.extend(list(np.unique(u)))
        vals = np.unique(vals)
        flags.append(set(vals) <= {0, 1})
    return flags

def remap_labels(name: str, y: pd.Series) -> np.ndarray:
    if name == "Covertype":
        return y.astype(int).to_numpy() - 1           # 1..7 -> 0..6
    if name == "Higgs":
        return y.astype(str).map({"b": 7, "s": 8}).to_numpy()
    if name == "HELOC":
        return y.astype(str).map({"Bad": 9, "Good": 10}).to_numpy()
    raise ValueError(name)

def winsorize_df(df: pd.DataFrame, cols: List[str], limits=(0.01, 0.99)) -> pd.DataFrame:
    if not cols:
        return df
    d = df.copy()
    lo, hi = limits
    for c in cols:
        x = d[c].astype(float)
        ql, qh = x.quantile(lo), x.quantile(hi)
        d[c] = x.clip(lower=ql, upper=qh)
    return d

def zscore_nan_safe_global(frame: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GLOBAL z-score: compute mu/sigma per column over ALL rows (ignoring NaNs),
    then return Z = (X - mu)/sigma, preserving NaNs.
    """
    X = frame[cols].astype(float).to_numpy()
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)  # avoid /0
    Z = (X - mu) / sigma
    return Z, mu, sigma

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Load, clean, dataset-aware FE
    raw_dfs: Dict[str, pd.DataFrame] = {}
    fe_dfs: Dict[str, pd.DataFrame] = {}
    info_log: Dict[str, dict] = {}
    for n, p in DATA_PATHS.items():
        df0, info = load_and_clean(n, p)
        info_log[n] = info
        raw_dfs[n] = df0.copy()          # for ownership inspection
        fe_dfs[n] = dataset_specific_fe(n, df0)

    # 2) Union feature space & ownership
    features = build_union_features(fe_dfs)
    ownership: Dict[str, Set[str]] = {f: set() for f in features}
    for n, d in fe_dfs.items():
        present = set(d.columns) - {TARGETS[n]}
        for f in present:
            ownership[f].add(n)

    # 3) Align + build masks + labels
    aligned: Dict[str, pd.DataFrame] = {}
    native_masks: Dict[str, pd.DataFrame] = {}
    struct_masks: Dict[str, pd.DataFrame] = {}
    labels: Dict[str, np.ndarray] = {}
    lengths: List[int] = []

    for n in ORDER:
        dfa = align(fe_dfs[n], features, TARGETS[n])
        aligned[n] = dfa

        present = set(fe_dfs[n].columns) - {TARGETS[n]}
        struct = pd.DataFrame(
            {f: (0.0 if f in present else 1.0) for f in features},
            index=dfa.index,
            dtype=np.float32,
        )
        native = dfa[features].isna().astype(np.float32) * (1.0 - struct.values)

        native_masks[n] = native
        struct_masks[n] = struct
        labels[n] = remap_labels(n, dfa[TARGETS[n]])
        lengths.append(len(dfa))

    # 4) Concatenate blocks
    X_concat = pd.concat([aligned[n][features] for n in ORDER], axis=0)
    native_concat = pd.concat([native_masks[n] for n in ORDER], axis=0).to_numpy(dtype=np.float32)
    struct_concat = pd.concat([struct_masks[n] for n in ORDER], axis=0).to_numpy(dtype=np.float32)
    y_all = np.concatenate([labels[n] for n in ORDER])
    ds_ids = np.concatenate([np.full(lengths[i], i, dtype=int) for i in range(len(ORDER))])
    ds_onehot = np.eye(len(ORDER), dtype=np.float32)[ds_ids]

    # 5) Split numeric vs binary
    is_binary = detect_binaries(aligned, features)
    binary_cols = [c for c, b in zip(features, is_binary) if b]
    numeric_cols = [c for c, b in zip(features, is_binary) if not b]

    # Optional winsorization per dataset (numerics only)
    if WINSORIZE and numeric_cols:
        parts = []
        start = 0
        for i, n in enumerate(ORDER):
            end = start + lengths[i]
            part = X_concat.iloc[start:end].copy()
            num_sub = [c for c in numeric_cols if c in part.columns]
            part[num_sub] = winsorize_df(part[num_sub], num_sub, limits=WINSOR_LIMITS)[num_sub]
            parts.append(part)
            start = end
        X_concat = pd.concat(parts, axis=0)

    # 6) GLOBAL scale numerics with NaN-robust z-score, then fill NaNs with 0 (neutral in z)
    if numeric_cols:
        X_num_z, mu, sigma = zscore_nan_safe_global(X_concat, numeric_cols)
        X_num = np.nan_to_num(X_num_z, nan=0.0).astype(np.float32)  # AE-safe
        # persist z-scaler params to reproduce at inference
        np.savez(
            ZSCALE_PARAMS_PATH,
            mu=mu.astype(np.float32),
            sigma=sigma.astype(np.float32),
            numeric_features=np.array(numeric_cols),
        )
    else:
        X_num = np.empty((len(X_concat), 0), dtype=np.float32)

    # 7) Binary imputation
    if binary_cols:
        X_bin_frame = X_concat[binary_cols].copy()
        if BINARY_IMPUTE_STRATEGY == "most_frequent":
            bin_imputer = SimpleImputer(strategy="most_frequent")
            X_bin = bin_imputer.fit_transform(X_bin_frame).astype(np.float32)
            joblib.dump(bin_imputer, IMPUTER_BIN_PATH)
        else:
            # constant_zero
            X_bin = X_bin_frame.fillna(0.0).to_numpy(dtype=np.float32)
            with open(IMPUTER_BIN_PATH.with_suffix(".txt"), "w") as f:
                f.write("Binary imputation: constant zero; no model artifact.")
    else:
        X_bin = np.empty((len(X_concat), 0), dtype=np.float32)

    # 8) Compose final matrix:
    #    [ numeric_z | binary | dataset_onehot | native_mask | struct_mask ]
    X_final = np.concatenate(
        [
            X_num,
            X_bin,
            ds_onehot,
            native_concat,
            struct_concat,
        ],
        axis=1,
    )

    # Block names
    block_names = (
        [f"{c}_z" for c in numeric_cols]
        + binary_cols
        + [f"is_{o.lower()}" for o in ORDER]
        + [f"{c}_native_mask" for c in features]
        + [f"{c}_struct_mask" for c in features]
    )

    # 9) Persist unified arrays
    np.savez(
        NPZ_PATH,
        X=X_final,
        y=y_all,
        features=np.array(features),
        numeric_features=np.array(numeric_cols),
        binary_features=np.array(binary_cols),
        dataset_ids=ds_ids,
        dataset_onehot_cols=np.array([f"is_{o.lower()}" for o in ORDER]),
        mask_native_cols=np.array([f"{c}_native_mask" for c in features]),
        mask_struct_cols=np.array([f"{c}_struct_mask" for c in features]),
        block_feature_names=np.array(block_names),
        order=np.array(ORDER),
    )

    # 10) Catalog for inspection
    catalog = []
    for c in numeric_cols:
        catalog.append({"feature": c, "block_out": f"{c}_z", "kind": "numeric_z"})
    for c in binary_cols:
        catalog.append({"feature": c, "block_out": c, "kind": "binary"})
    for c in [f"is_{o.lower()}" for o in ORDER]:
        catalog.append({"feature": c, "block_out": c, "kind": "dataset_onehot"})
    for c in [f"{c}_native_mask" for c in features]:
        catalog.append({"feature": c, "block_out": c, "kind": "missing_mask_native"})
    for c in [f"{c}_struct_mask" for c in features]:
        catalog.append({"feature": c, "block_out": c, "kind": "missing_mask_struct"})
    pd.DataFrame(catalog).to_csv(FEATURE_CATALOG_PATH, index=False)

    # 11) Meta & logging
    meta = {
        "npz_path": str(NPZ_PATH),
        "zscaler_params": str(ZSCALE_PARAMS_PATH),
        "binary_imputer": str(IMPUTER_BIN_PATH),
        "label_mapping": {
            "Covertype": "1..7 -> 0..6",
            "Higgs": "b -> 7, s -> 8",
            "HELOC": "Bad -> 9, Good -> 10",
        },
        "notes": {
            "covertype_fe": {
                "Aspect_cos": "kept",
                "Elev_minus_VertHydro": "kept",
                "Aspect": "dropped",
                "Hillshade_mean": "excluded",
            },
            "heloc": {
                "nmar_codes": [-7, -8, -9],
                "policy": "keep rows; convert codes -> NaN; numeric NaN -> 0 after global z-scaling",
                "rows_with_any_code": int(meta.get("notes", {}).get("heloc", {}).get("rows_with_any_code", 0)) if False else 0,
            },
            "masks": {
                "native": "<feat>_native_mask (exists & NaN)",
                "struct": "<feat>_struct_mask (feature absent in dataset)",
            },
            "scaling_imputation": {
                "numeric": "GLOBAL z-score (nan-robust), then NaN->0 in z-space",
                "binary": f"{'most_frequent' if BINARY_IMPUTE_STRATEGY=='most_frequent' else 'constant_zero'}",
            },
            "dataset_onehot": "conditioning signal for unified model",
            "winsorize": {"enabled": WINSORIZE, "limits": WINSOR_LIMITS},
        },
        "feature_ownership": {f: sorted(list(ownership[f])) for f in features},
        "source_info": {
            "heloc_rows_kept_with_codes": int(info_log.get("HELOC", {}).get("heloc_rows_kept_with_codes", 0)),
            "heloc_total_rows": int(info_log.get("HELOC", {}).get("heloc_total_rows", 0)),
            "heloc_code_counts": info_log.get("HELOC", {}).get("heloc_code_counts", {}),
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", NPZ_PATH)
    print("Saved z-scaler params:", ZSCALE_PARAMS_PATH)
    print("Saved binary imputer or note:", IMPUTER_BIN_PATH)
    print("Saved catalog:", FEATURE_CATALOG_PATH)
    print("Saved meta:", META_PATH)
    print("Final X shape:", X_final.shape, "y shape:", y_all.shape)

if __name__ == "__main__":
    main()
