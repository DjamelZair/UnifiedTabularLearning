# train_eval_xgboost_from_latents_v6.py
# XGBoost baseline on AE latents (v6)
# - Loads Z (latents), y (union labels 0..10), ds_ids from MERGED_DIR
# - Adds dataset one-hot as extra features (optional)
# - Stratified TEST hold-out (by dataset & local class)
# - Early-stopped training with a stratified Train/Val split on TrainVal
# - Saves model -> models/, metrics -> results/metrics, plots -> results/plots

import sys, os, platform, math, warnings, json, time, random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import importlib.util

# ---- discover project paths (same as other scripts) ----
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from tabster_paths import PROJECT_ROOT, MERGED_DIR, MODELS_DIR, SCRIPTS_DIR  # type: ignore

# ---- plotting helpers (reuse your scripts/tabster_viz.py) ----
try:
    from scripts.tabster_viz import (
        configure_matplotlib, make_saver,
        make_confusion, make_multiclass_roc
    )
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location("tabster_viz", str(SCRIPTS_DIR / "tabster_viz.py"))
    tv = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tv)
    configure_matplotlib = tv.configure_matplotlib
    make_saver = tv.make_saver
    make_confusion = tv.make_confusion
    make_multiclass_roc = tv.make_multiclass_roc

# =========================
# Config
# =========================
SEED = 42
ORDER = ["Covertype", "Higgs", "HELOC"]
BLOCKS = {"Covertype": (0, 7), "Higgs": (7, 2), "HELOC": (9, 2)}
N_CLASSES_UNION = 11

INCLUDE_DS_ONEHOT = True      # add dataset one-hot to Z
TEST_RATIO        = 0.10      # stratified by (dataset, local class)
VAL_RATIO         = 0.20      # inside TrainVal, for early stopping

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_TAG = "v6_xgb"
MODEL_PATH = MODELS_DIR / "xgb_unified_latents_v6.json"
CSV_TEST_SUMMARY = METRICS_DIR / "xgb_internal_test_summary_v6.csv"
CSV_TEST_PRED    = METRICS_DIR / "xgb_internal_test_predictions_v6.csv"

# Reasonable defaults for dense latents
XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=N_CLASSES_UNION,
    eval_metric="mlogloss",
    tree_method="hist",      # fast on CPU
    predictor="cpu_predictor",
    eta=0.10,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=SEED,
)

NUM_BOOST_ROUND = 800
EARLY_STOPPING  = 50
VERBOSE_EVAL    = 50   # set 0 to silence

# =========================
# Small helpers
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def to_local(y_unified_subset, ds_name):
    start, _ = BLOCKS[ds_name]
    return y_unified_subset - start

def softmax_np(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    s = e.sum(axis=1, keepdims=True)
    return e / np.maximum(s, 1e-12)

def make_strat_label(y_u, ds_ids, order):
    lab = np.zeros_like(y_u, dtype=int)
    for i, name in enumerate(order):
        m = (ds_ids == i)
        yloc = to_local(y_u[m], name)
        lab[m] = i * 32 + yloc  # 32 > max classes per dataset
    return lab

def stratified_test_holdout(ds_ids, y, order, test_ratio=TEST_RATIO, seed=SEED):
    rng = np.random.default_rng(seed)
    idx_test, idx_rest = [], []
    for i, ds_name in enumerate(order):
        rows = np.where(ds_ids == i)[0]
        y_local = to_local(y[rows], ds_name)
        classes = np.unique(y_local)
        for c in classes:
            rc = rows[y_local == c]
            rng.shuffle(rc)
            n_test = max(1, int(round(len(rc) * test_ratio)))
            idx_test.extend(rc[:n_test])
            idx_rest.extend(rc[n_test:])
    return np.array(idx_rest), np.array(idx_test)

def evaluate_union_and_per_ds(order, y_u_true, y_u_pred, proba_union, ds_ids):
    # union
    union_acc = accuracy_score(y_u_true, y_u_pred)
    union_f1  = f1_score(y_u_true, y_u_pred, average="macro")

    # per-dataset
    per_ds = {}
    for i, name in enumerate(order):
        m = (ds_ids == i)
        if not np.any(m):
            continue
        y_true_local = to_local(y_u_true[m], name)
        y_pred_local = to_local(y_u_pred[m], name)
        per_ds[name] = {
            "acc": float(accuracy_score(y_true_local, y_pred_local)),
            "f1_macro": float(f1_score(y_true_local, y_pred_local, average="macro")),
        }

    f1s  = [d["f1_macro"] for _, d in per_ds.items()]
    accs = [d["acc"] for _, d in per_ds.items()]
    mean_f1  = float(np.mean(f1s)) if len(f1s) else union_f1
    mean_acc = float(np.mean(accs)) if len(accs) else union_acc

    return dict(union_acc=float(union_acc),
                union_f1=float(union_f1),
                mean_acc=mean_acc,
                mean_f1=mean_f1,
                per_ds=per_ds)

# =========================
# Data loading
# =========================
def load_latents_and_meta():
    z_path = MERGED_DIR / "trainval_latents_v6.npy"
    meta_p = MERGED_DIR / "unified_data_v6.npz"
    if not z_path.exists() or not meta_p.exists():
        raise FileNotFoundError("Missing latents or meta npz. Run AE + merge first.")
    Z = np.load(z_path).astype(np.float32)
    meta = np.load(meta_p, allow_pickle=True)
    y = meta["y"].astype(int)
    ds_ids = meta["dataset_ids"].astype(int)
    order = [str(x) for x in meta["order"]] if "order" in meta.files else ORDER
    return Z, y, ds_ids, order

def augment_with_ds_onehot(Z, ds_ids, order, include=INCLUDE_DS_ONEHOT):
    if not include:
        return Z
    onehot = np.eye(len(order), dtype=np.float32)[ds_ids]
    return np.hstack([Z, onehot])

# =========================
# Main
# =========================
def main():
    print("\n" + "="*80 + "\nENVIRONMENT\n" + "="*80, flush=True)
    print(f"[Python] {platform.python_version()}", flush=True)
    print(f"[XGBoost] {xgb.__version__}", flush=True)

    set_seed()
    configure_matplotlib()
    save_png = make_saver(PLOTS_DIR, tag=PLOT_TAG)

    # ---- Load data
    Z, y, ds_ids, order = load_latents_and_meta()
    X = augment_with_ds_onehot(Z, ds_ids, order, include=INCLUDE_DS_ONEHOT)

    # ---- Stratified hold-out TEST (by dataset & local class)
    idx_tv, idx_te = stratified_test_holdout(ds_ids, y, order, TEST_RATIO, SEED)
    Xtv, ytv, dstv = X[idx_tv], y[idx_tv], ds_ids[idx_tv]
    Xte, yte, dste = X[idx_te], y[idx_te], ds_ids[idx_te]

    # ---- Stratified Train/Val split on TrainVal (for early stopping)
    strat_tv = make_strat_label(ytv, dstv, order)
    tr_idx, va_idx = next(StratifiedKFold(n_splits=int(1/VAL_RATIO), shuffle=True, random_state=SEED)
                          .split(Xtv, strat_tv))
    Xtr, ytr, dstr = Xtv[tr_idx], ytv[tr_idx], dstv[tr_idx]
    Xva, yva, dsva = Xtv[va_idx], ytv[va_idx], dstv[va_idx]

    # ---- Train with early stopping
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval   = xgb.DMatrix(Xva, label=yva)
    evals  = [(dtrain, "train"), (dval, "val")]

    print("\n" + "="*80 + "\nTRAINING (XGBoost)\n" + "="*80, flush=True)
    bst = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        evals=evals,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=VERBOSE_EVAL
    )

    # ---- Evaluate on TEST
    print("\n" + "="*80 + "\nTEST EVALUATION\n" + "="*80, flush=True)
    dtest = xgb.DMatrix(Xte)
    proba = bst.predict(dtest)
    if proba.ndim == 1:   # safety if binary config sneaks in
        proba = np.vstack([1.0 - proba, proba]).T
    y_pred = proba.argmax(1)

    res = evaluate_union_and_per_ds(order, yte, y_pred, proba, dste)
    print(f"union_acc={res['union_acc']:.4f} | union_f1={res['union_f1']:.4f} | "
          f"mean_acc={res['mean_acc']:.4f} | mean_f1={res['mean_f1']:.4f}")
    for ds, d in res["per_ds"].items():
        print(f"  {ds}: acc={d['acc']:.4f}  f1_macro={d['f1_macro']:.4f}")

    # ---- Plots (union)
    make_confusion(save_png, yte, y_pred, classes=range(N_CLASSES_UNION),
                   title="XGBoost Unified • Confusion (TEST)")
    make_multiclass_roc(save_png, yte, proba, n_classes=N_CLASSES_UNION,
                        title="XGBoost Unified • ROC (TEST)")

    # ---- Save artifacts
    bst.save_model(str(MODEL_PATH))
    print(f"Saved model -> {MODEL_PATH}")

    # CSV metrics
    rows = [
        ["union_acc",  f"{res['union_acc']:.6f}"],
        ["union_f1",   f"{res['union_f1']:.6f}"],
        ["mean_acc",   f"{res['mean_acc']:.6f}"],
        ["mean_f1",    f"{res['mean_f1']:.6f}"],
    ]
    for ds, d in res["per_ds"].items():
        rows.append([f"{ds}_acc",      f"{d['acc']:.6f}"])
        rows.append([f"{ds}_f1_macro", f"{d['f1_macro']:.6f}"])
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(CSV_TEST_SUMMARY, index=False)
    print(f"Saved test summary -> {CSV_TEST_SUMMARY}")

    # CSV predictions
    pd.DataFrame({
        "dataset_id": dste,
        "y_true_unified": yte.astype(int),
        "y_pred_unified": y_pred.astype(int),
    }).to_csv(CSV_TEST_PRED, index=False)
    print(f"Saved test predictions -> {CSV_TEST_PRED}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
