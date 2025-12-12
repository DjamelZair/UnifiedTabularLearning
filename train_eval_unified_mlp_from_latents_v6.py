# TABSTER2/train_eval_unified_mlp_from_latents_v6.py
# Single-trial training with fixed best hyperparameters only.
# Plots only after training (best checkpoint). Progress bar + star logs only.

import sys, os, platform, itertools, math, warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import importlib.util

# ---- ensure we can import tabster_paths from the project root ----
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from tabster_paths import PROJECT_ROOT, MERGED_DIR, MODELS_DIR, SCRIPTS_DIR, DATA_DIR  # type: ignore

# ---- plotting helpers (lives in PROJECT_ROOT/scripts) ----
try:
    from scripts.tabster_viz import (
        configure_matplotlib, make_saver, plot_dataset_size_imbalance,
        make_confusion, make_multiclass_roc, make_binary_roc,
        make_pred_vs_true_counts, make_binary_prob_by_true
    )
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location("tabster_viz", str(SCRIPTS_DIR / "tabster_viz.py"))
    tv = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tv)
    configure_matplotlib = tv.configure_matplotlib
    make_saver = tv.make_saver
    plot_dataset_size_imbalance = tv.plot_dataset_size_imbalance
    make_confusion = tv.make_confusion
    make_multiclass_roc = tv.make_multiclass_roc
    make_binary_roc = tv.make_binary_roc
    make_pred_vs_true_counts = tv.make_pred_vs_true_counts
    make_binary_prob_by_true = tv.make_binary_prob_by_true

# Configurations
SEED = 42
N_CLASSES = 11
ORDER_DEFAULT = ["Covertype", "Higgs", "HELOC"]
BLOCKS = {"Covertype": (0, 7), "Higgs": (7, 2), "HELOC": (9, 2)}

# Split: 70/20/10 for hyperparam validation & internal test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

# Training controls
EPOCHS      = 30 # one full pass through the entire training dataset
PATIENCE    = 8 # used for early stopping to prevent overfitting, safes training time, stops when training reaches a plateau
USE_AMP     = True
NUM_WORKERS = 2 # Data loading parallelism

HP_DEFAULT = dict(HIDDEN=256, DROPOUT=0.0, LR=1e-3, WEIGHT_DECAY=0.0, BATCH_SIZE=256)

# output locations
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"        # figures here
METRICS_DIR = RESULTS_DIR / "metrics"      # csv metrics here

# keep checkpoints here unless you want them moved as well
OUTDIR = RESULTS_DIR / "results_singlehead"
CKPT   = OUTDIR / "unified_mlp_singlehead_best_v6.pt"
TXT_BEST = OUTDIR / "best_config_v6.txt"

CSV_INT_TEST_SUMMARY = METRICS_DIR / "internal_test_summary_v6.csv"
CSV_INT_TEST_PRED    = METRICS_DIR / "internal_test_predictions_v6.csv"
CSV_TRAIN_CURVE      = METRICS_DIR / "training_curve_v6.csv"


def load_best_hp(txt_path: Path, fallback: dict) -> dict:
    """Load HPs from best_config file if present; otherwise return fallback defaults."""
    hp = dict(fallback)
    if not txt_path.exists():
        return hp
    for line in txt_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().upper()
        if key not in hp:
            continue
        v = val.strip()
        try:
            hp[key] = int(v) if v.isdigit() else float(v)
        except ValueError:
            hp[key] = v
    return hp

HP = load_best_hp(TXT_BEST, HP_DEFAULT)

# =========================
# Small helpers (quiet)
# =========================
def p_header(title): print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80, flush=True)
def p_kv(k, v): print(f"[{k}] {v}", flush=True)
def p_counts(label, arr):
    arr = np.asarray(arr); uniq, cnt = np.unique(arr, return_counts=True)
    print(f"{label} -> " + ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt)]), flush=True)

def set_seed(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def softmax_np(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a); s = e.sum(axis=1, keepdims=True)
    return e / np.maximum(s, 1e-12)

def to_local(y_unified_subset, ds_name):
    start, _ = BLOCKS[ds_name]; return y_unified_subset - start

def class_weights_union(y_train_unified):
    cnt = np.bincount(y_train_unified, minlength=N_CLASSES)
    cnt = np.maximum(cnt, 1)
    w = 1.0 / np.sqrt(cnt)
    return torch.tensor(w / w.mean(), dtype=torch.float32)

def make_balanced_sampler_by_ds_and_class(ds_ids_subset, y_unified_subset, order):
    # weight by 1/freq(dataset,class)
    freq = {}
    for i, name in enumerate(order):
        mask = (ds_ids_subset == i)
        yloc = to_local(y_unified_subset[mask], name)
        vals, cnts = np.unique(yloc, return_counts=True)
        for v, c in zip(vals, cnts): freq[(i, int(v))] = int(c)
    weights = np.empty(len(ds_ids_subset), dtype=np.float32)
    for idx in range(len(ds_ids_subset)):
        d = int(ds_ids_subset[idx]); ds_name = order[d]
        c = int(to_local(np.array([y_unified_subset[idx]]), ds_name)[0])
        weights[idx] = 1.0 / max(freq[(d, c)], 1)
    return WeightedRandomSampler(weights, num_samples=len(ds_ids_subset), replacement=True)

# latents loader
def load_trainval_latents_and_meta():
    z_path = MERGED_DIR / "trainval_latents_v6.npy"
    meta_p = MERGED_DIR / "unified_data_v6.npz"
    if not z_path.exists() or not meta_p.exists():
        raise FileNotFoundError("Missing latents or meta npz. Run AE + merge first.")
    Z = np.load(z_path).astype(np.float32)
    meta = np.load(meta_p, allow_pickle=True)
    y = meta["y"].astype(int)
    ds_ids = meta["dataset_ids"].astype(int)
    order = [str(x) for x in meta["order"]] if "order" in meta.files else ORDER_DEFAULT
    return Z, y, ds_ids, order

def make_train_val_test_split(Z, y, ds_ids, order, seed=SEED):
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []
    for i, ds_name in enumerate(order):
        rows = np.where(ds_ids == i)[0]
        y_local = to_local(y[rows], ds_name)
        for c in np.unique(y_local):
            rows_c = rows[y_local == c]
            rng.shuffle(rows_c)
            n = len(rows_c)
            n_train = int(round(n * TRAIN_RATIO))
            n_val   = int(round(n * VAL_RATIO))
            n_test  = n - n_train - n_val
            idx_train.extend(rows_c[:n_train])
            idx_val.extend(rows_c[n_train:n_train+n_val])
            idx_test.extend(rows_c[n_train+n_val:])
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

# data and model
class UnifiedDataset(Dataset):
    def __init__(self, X, y_unified, ds_ids): self.X=X; self.y=y_unified; self.ds=ds_ids
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.ds[i]

class Trunk(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class SingleHead(nn.Module):
    def __init__(self, hidden, n_classes=N_CLASSES):
        super().__init__(); self.out = nn.Linear(hidden, n_classes)
    def forward(self, h): return self.out(h)

# evaluation and plots
def slice_logits_for_dataset(logits, ds_name):
    start, length = BLOCKS[ds_name]; return logits[:, start:start+length]

def evaluate_union_and_slices(cfg, order, trunk, head, X, y_u, ds_ids, device,
                              make_plots=False, name_prefix=""):
    trunk.eval(); head.eval()
    Xv = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        H = trunk(Xv)
        logits = head(H).cpu().numpy()
    proba = softmax_np(logits)
    y_pred_u = proba.argmax(1)

    if y_u is not None:
        acc_union = accuracy_score(y_u, y_pred_u)
        f1_union = f1_score(y_u, y_pred_u, average="macro")
    else:
        acc_union = float("nan"); f1_union=float("nan")

    per_ds = {}
    for ds_idx, ds_name in enumerate(order):
        m = (ds_ids == ds_idx)
        if not np.any(m): continue
        log_sub = slice_logits_for_dataset(logits[m], ds_name)
        prob_sub = softmax_np(log_sub)
        y_pred_local = prob_sub.argmax(1)
        if y_u is not None:
            y_true_local = to_local(y_u[m], ds_name)
            acc = accuracy_score(y_true_local, y_pred_local)
            f1m = f1_score(y_true_local, y_pred_local, average="macro")
            per_ds[ds_name] = {"acc": float(acc), "f1_macro": float(f1m)}
        else:
            per_ds[ds_name] = {"acc": float("nan"), "f1_macro": float("nan")}

        if make_plots and y_u is not None and name_prefix:
            if ds_name == "Covertype":
                make_confusion(cfg.save_png, y_true_local, y_pred_local, classes=range(7),
                               title=f"{name_prefix} Covertype Confusion Matrix")
                make_multiclass_roc(cfg.save_png, y_true_local, prob_sub, n_classes=7,
                                    title=f"{name_prefix} Covertype ROC Curve")
                make_pred_vs_true_counts(cfg.save_png, y_true_local, y_pred_local, classes=range(7),
                                         title=f"{name_prefix} Covertype Predictions versus True")
            else:
                pos_prob = prob_sub[:, 1]
                make_confusion(cfg.save_png, y_true_local, y_pred_local, classes=[0,1],
                               title=f"{name_prefix} {ds_name} Confusion Matrix")
                make_binary_roc(cfg.save_png, y_true_local, pos_prob, title=f"{name_prefix} {ds_name} ROC Curve")
                make_pred_vs_true_counts(cfg.save_png, y_true_local, y_pred_local, classes=[0,1],
                                         title=f"{name_prefix} {ds_name} Predictions versus True")
                make_binary_prob_by_true(cfg.save_png, y_true_local, pos_prob,
                                         title=f"{name_prefix} {ds_name} p(pos) by True Class")

    if make_plots and y_u is not None and name_prefix:
        make_confusion(cfg.save_png, y_u, y_pred_u, classes=range(N_CLASSES),
                       title=f"{name_prefix} Unified Confusion Matrix")
        make_multiclass_roc(cfg.save_png, y_u, proba, n_classes=N_CLASSES,
                            title=f"{name_prefix} Unified ROC Curve")

    f1s  = [per_ds[k]["f1_macro"] for k in per_ds] or [f1_union]
    accs = [per_ds[k]["acc"] for k in per_ds]       or [acc_union]
    return {
        "union_acc": float(acc_union),
        "union_f1": float(f1_union),
        "mean_acc": float(np.nanmean(accs)),
        "mean_f1": float(np.nanmean(f1s)),
        "per_ds": per_ds,
        "y_pred_unified": y_pred_u,
        "proba": proba,
    }

# train, single trial
def train_single(cfg, hp, Xtr, ytr, dstr, Xva, yva, dsva, order, device,
                 epochs, patience):
    set_seed()
    train_ds = UnifiedDataset(Xtr, ytr, dstr)
    sampler  = make_balanced_sampler_by_ds_and_class(dstr, ytr, order)
    train_ld = DataLoader(train_ds, batch_size=hp["BATCH_SIZE"], sampler=sampler,
                          drop_last=False, num_workers=NUM_WORKERS)

    input_dim = Xtr.shape[1]
    trunk = Trunk(input_dim, hidden=hp["HIDDEN"], dropout=hp["DROPOUT"]).to(device)
    head  = SingleHead(hidden=hp["HIDDEN"], n_classes=N_CLASSES).to(device)

    w = class_weights_union(ytr).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(head.parameters()),
                           lr=hp["LR"], weight_decay=hp["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    # AMP scaler (new API with fallback)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.startswith("cuda"))
        autocast_ctx = lambda: torch.amp.autocast(device_type="cuda", enabled=USE_AMP and device.startswith("cuda"))
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler, autocast
        scaler = CudaGradScaler(enabled=USE_AMP and device.startswith("cuda"))
        autocast_ctx = lambda: autocast(enabled=USE_AMP and device.startswith("cuda"))

    best_score = -math.inf
    best_state = None
    best_epoch = 0
    no_improve = 0

    # per-epoch logging to metrics file
    log_rows = []

    ep_iter = tqdm(range(1, epochs + 1), desc=f"Trial 1", leave=True)
    for epoch in ep_iter:
        trunk.train(); head.train()
        for Xb, yb, _dsb in train_ld:
            Xb = Xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = head(trunk(Xb))
                loss = nn.functional.cross_entropy(logits, yb, weight=w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        res = evaluate_union_and_slices(cfg, order, trunk, head, Xva, yva, dsva, device, make_plots=False)
        score = res["mean_f1"]
        sched.step(score)

        ep_iter.set_postfix({"mean_f1": f"{score:.4f}",
                             "cov": f"{res['per_ds'].get('Covertype',{}).get('f1_macro',float('nan')):.4f}",
                             "hig": f"{res['per_ds'].get('Higgs',{}).get('f1_macro',float('nan')):.4f}",
                             "hel": f"{res['per_ds'].get('HELOC',{}).get('f1_macro',float('nan')):.4f}"})

        # collect per-epoch metrics
        log_rows.append({
            "epoch": int(epoch),
            "mean_f1": float(score),
            "cov_f1": float(res["per_ds"].get("Covertype",{}).get("f1_macro", np.nan)),
            "hig_f1": float(res["per_ds"].get("Higgs",{}).get("f1_macro", np.nan)),
            "hel_f1": float(res["per_ds"].get("HELOC",{}).get("f1_macro", np.nan)),
        })

        if score > best_score:
            best_score = score
            best_state = {
                "trunk": trunk.state_dict(),
                "head":  head.state_dict(),
                "config": {
                    "HIDDEN": hp["HIDDEN"], "DROPOUT": hp["DROPOUT"],
                    "LR": hp["LR"], "WEIGHT_DECAY": hp["WEIGHT_DECAY"],
                    "BATCH_SIZE": hp["BATCH_SIZE"], "EPOCHS": epoch, "SEED": SEED,
                }
            }
            best_epoch = epoch
            no_improve = 0
            tqdm.write(
                f"★ Trial 1 epoch {epoch} new best mean_f1={best_score:.4f}  "
                f"[Cov {res['per_ds'].get('Covertype',{}).get('f1_macro',float('nan')):.4f} | "
                f"HIG {res['per_ds'].get('Higgs',{}).get('f1_macro',float('nan')):.4f} | "
                f"HEL {res['per_ds'].get('HELOC',{}).get('f1_macro',float('nan')):.4f}]"
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"Early stopping Trial 1 at epoch {epoch}")
                break

    # write per-epoch training curve
    if log_rows:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_rows).to_csv(CSV_TRAIN_CURVE, index=False)

    return best_state, best_epoch, best_score

# main
def main():
    p_header("ENVIRONMENT")
    p_kv("Python", platform.python_version())
    p_kv("Torch", torch.__version__)
    p_kv("CUDA available", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p_kv("Using device", device)

    set_seed()

    # ensure directories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    save_png = make_saver(PLOTS_DIR, tag="v6")

    Z, y, ds_ids, order = load_trainval_latents_and_meta()
    idx_tr, idx_va, idx_te = make_train_val_test_split(Z, y, ds_ids, order)
    Xtr, Xva, Xte = Z[idx_tr], Z[idx_va], Z[idx_te]
    ytr, yva, yte = y[idx_tr], y[idx_va], y[idx_te]
    dstr, dsva, dste = ds_ids[idx_tr], ds_ids[idx_va], ds_ids[idx_te]

    # ---- Train single trial with fixed HPs
    state, best_epoch, score = train_single(SimpleNamespace(save_png=save_png),
                                            HP, Xtr, ytr, dstr, Xva, yva, dsva, order,
                                            device, epochs=EPOCHS, patience=PATIENCE)

    # Save checkpoint + best config
    torch.save(state, CKPT)
    with open(TXT_BEST, "w") as f:
        f.write("Best HP configuration (v6)\n")
        for k, v in HP.items(): f.write(f"{k}: {v}\n")
        f.write(f"\nBest epoch: {best_epoch}\n")
    print(f"Saved best checkpoint to {CKPT}")
    print(f"Wrote best config to {TXT_BEST}")

    # plots only after training
    trunk = Trunk(Xtr.shape[1], hidden=HP["HIDDEN"], dropout=HP["DROPOUT"]).to(device)
    head  = SingleHead(hidden=HP["HIDDEN"], n_classes=N_CLASSES).to(device)
    trunk.load_state_dict(state["trunk"]); head.load_state_dict(state["head"])
    evaluate_union_and_slices(SimpleNamespace(save_png=save_png),
                              order, trunk, head, Xva, yva, dsva, device,
                              make_plots=True, name_prefix="VALIDATION")
    p_header("INTERNAL TEST EVALUATION")
    res_te = evaluate_union_and_slices(SimpleNamespace(save_png=make_saver(PLOTS_DIR, tag="v6")),
                                       order, trunk, head, Xte, yte, dste, device,
                                       make_plots=True, name_prefix="INTERNAL TEST")

    # Save internal test CSVs
    pred_df = pd.DataFrame({
        "dataset_id": dste,
        "y_true_unified": yte.astype(int),
        "y_pred_unified": res_te["y_pred_unified"].astype(int),
    })
    pred_df.to_csv(CSV_INT_TEST_PRED, index=False)
    rows = [
        ["union_acc",  f"{res_te['union_acc']:.6f}"],
        ["union_f1",   f"{res_te['union_f1']:.6f}"],
        ["mean_acc",   f"{res_te['mean_acc']:.6f}"],
        ["mean_f1",    f"{res_te['mean_f1']:.6f}"],
    ]
    for ds, d in res_te["per_ds"].items():
        rows.append([f"{ds}_acc",      f"{d['acc']:.6f}"])
        rows.append([f"{ds}_f1_macro", f"{d['f1_macro']:.6f}"])
    pd.DataFrame(rows, columns=["metric","value"]).to_csv(CSV_INT_TEST_SUMMARY, index=False)
    print(f"Saved internal test predictions to {CSV_INT_TEST_PRED}")
    print(f"Saved internal test summary to {CSV_INT_TEST_SUMMARY}")
    print(f"Saved training curve to {CSV_TRAIN_CURVE}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
