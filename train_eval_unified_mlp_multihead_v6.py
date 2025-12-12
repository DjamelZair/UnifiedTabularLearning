# TABSTER2/train_eval_unified_mlp_multihead_kfold_v7.py
# Robust multi-head MLP on AE latents with:
# - Stratified TEST hold-out (by dataset & local class)
# - Stratified K-Fold CV on TrainVal for HP selection
# - Early stopping per-fold; final refit on all TrainVal for fixed epochs = median(best_epoch)
# - Balanced sampling + class-weighted losses per dataset head
# - NO plots during training; ONLY once for final (Val-on-refit + Test)
# - Leak-tight: Test unseen until the very end; no val used during refit

import sys, os, platform, math, warnings, json, time, random
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import importlib.util

# ---- discover project paths ----
_THIS = Path(__file__).resolve()
for cand in [_THIS.parent, _THIS.parent.parent, *_THIS.parents]:
    if (cand / "tabster_paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from tabster_paths import PROJECT_ROOT, MERGED_DIR, SCRIPTS_DIR  # type: ignore

# ---- plotting helpers (in scripts/tabster_viz.py) ----
try:
    from scripts.tabster_viz import (
        configure_matplotlib, make_saver,
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
    make_confusion = tv.make_confusion
    make_multiclass_roc = tv.make_multiclass_roc
    make_binary_roc = tv.make_binary_roc
    make_pred_vs_true_counts = tv.make_pred_vs_true_counts
    make_binary_prob_by_true = tv.make_binary_prob_by_true

# =========================
# Global Config
# =========================
SEED = 42
ORDER = ["Covertype", "Higgs", "HELOC"]
BLOCKS = {"Covertype": (0, 7), "Higgs": (7, 2), "HELOC": (9, 2)}
N_CLASSES_UNION = 11

# Hold-out for true generalization
TEST_RATIO  = 0.10  # stratified by (dataset,local_class)

# CV and training runtime
K_FOLDS           = 2    # auto-reduced if a stratum is too small
EPOCHS_MAX        = 1
PATIENCE          = 8
WARMUP_EPOCHS     = 3
GRAD_CLIP_NORM    = 1.0
USE_AMP           = True
NUM_WORKERS       = 0

# Print per-epoch validation metrics (no plots)
PRINT_EPOCH_METRICS = True
EPOCH_LOG_EVERY     = 1

PLOT_AFTER_FINAL_ONLY = True

# Base & grid
BASE_HP = dict(HIDDEN=256, DROPOUT=0.0, LR=1e-3, WEIGHT_DECAY=0.0, BATCH_SIZE=256)
HP_GRID = [
    BASE_HP,
    dict(HIDDEN=256, DROPOUT=0.10, LR=1e-3,  WEIGHT_DECAY=0.0,  BATCH_SIZE=256),
    dict(HIDDEN=384, DROPOUT=0.10, LR=7e-4,  WEIGHT_DECAY=1e-5, BATCH_SIZE=256),
    dict(HIDDEN=192, DROPOUT=0.00, LR=1.2e-3,WEIGHT_DECAY=0.0,  BATCH_SIZE=256),
]

MODEL_TAG = "[Multi-Head MLP]"

# ---- outputs
PLOTS  = PROJECT_ROOT / "plots"
OUTDIR = PROJECT_ROOT / "results_latents_v7"
CKPT   = OUTDIR / "unified_mlp_multihead_best_v7.pt"
TXT_BEST = OUTDIR / "best_config_multihead_v7.txt"
CSV_INT_TEST_SUMMARY = OUTDIR / "internal_test_summary_multihead_v7.csv"
CSV_INT_TEST_PRED    = OUTDIR / "internal_test_predictions_multihead_v7.csv"
CSV_TRIALS_LOG       = OUTDIR / "multihead_grid_results_v7.csv"
CSV_TRIALS_FOLDS     = OUTDIR / "multihead_grid_folds_v7.csv"

# =========================
# Utility
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def softmax_np(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    s = e.sum(axis=1, keepdims=True)
    return e / np.maximum(s, 1e-12)

def to_local(y_unified_subset, ds_name):
    start, _ = BLOCKS[ds_name]
    return y_unified_subset - start

def make_strat_label(y_u, ds_ids, order):
    lab = np.zeros_like(y_u, dtype=int)
    for i, name in enumerate(order):
        m = (ds_ids == i)
        yloc = to_local(y_u[m], name)
        lab[m] = i * 32 + yloc  # 32 > max classes per dataset
    return lab

def get_loader_kwargs(device):
    pin = device == "cuda"
    persist = False if NUM_WORKERS == 0 else (device == "cuda")
    return dict(num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=persist)

# =========================
# Data I/O
# =========================
def load_trainval_latents_and_meta():
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

def auto_adjust_kfold(trainval_idx, y, ds_ids, order, requested_k=K_FOLDS):
    lab = make_strat_label(y[trainval_idx], ds_ids[trainval_idx], order)
    _, cnts = np.unique(lab, return_counts=True)
    min_stratum = int(cnts.min()) if len(cnts) else 1
    K = max(2, min(requested_k, min_stratum))
    return K

# =========================
# Dataset & Sampler
# =========================
class UnifiedDataset(Dataset):
    def __init__(self, X, y_unified, ds_ids):
        self.X = X
        self.y = y_unified
        self.ds = ds_ids
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.ds[i]

def make_balanced_sampler_by_ds_and_class(ds_ids_subset, y_unified_subset, order):
    # weights = 1 / freq for (dataset, local_class)
    freq = {}
    for i, name in enumerate(order):
        mask = (ds_ids_subset == i)
        if not np.any(mask): 
            continue
        yloc = to_local(y_unified_subset[mask], name)
        vals, cnts = np.unique(yloc, return_counts=True)
        for v, c in zip(vals, cnts):
            freq[(i, int(v))] = int(c)
    weights = np.empty(len(ds_ids_subset), dtype=np.float32)
    for idx in range(len(ds_ids_subset)):
        d = int(ds_ids_subset[idx])
        name = order[d]
        c = int(to_local(np.array([y_unified_subset[idx]]), name)[0])
        weights[idx] = 1.0 / max(freq.get((d, c), 1), 1)
    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(ds_ids_subset), replacement=True)

def class_weights_per_dataset(y_unified, ds_ids, order):
    out = {}
    for i, name in enumerate(order):
        m = (ds_ids == i)
        if not np.any(m):
            continue
        yloc = to_local(y_unified[m], name)
        n_classes = BLOCKS[name][1]
        cnt = np.bincount(yloc, minlength=n_classes).astype(float)
        cnt = np.maximum(cnt, 1.0)
        w = 1.0 / np.sqrt(cnt)
        out[name] = torch.tensor(w / w.mean(), dtype=torch.float32)
    return out

def _safe_weight(w_map, name, device):
    t = w_map.get(name)
    if t is None:
        n = BLOCKS[name][1]
        return torch.ones(n, device=device)
    return t.to(device)

# =========================
# Model (trunk + 3 heads)
# =========================
class Trunk(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Heads(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.cov = nn.Linear(hidden, 7)
        self.hig = nn.Linear(hidden, 2)
        self.hel = nn.Linear(hidden, 2)
    def forward(self, h):
        return {"Covertype": self.cov(h), "Higgs": self.hig(h), "HELOC": self.hel(h)}

# =========================
# Evaluation (NO plots during training)
# =========================
@torch.no_grad()
def evaluate_multi(cfg, order, trunk, heads, X, y_u, ds_ids, device,
                   make_plots=False, name_prefix=""):
    trunk.eval()
    heads.eval()
    Xv = torch.tensor(X, dtype=torch.float32, device=device)
    H  = trunk(Xv)

    per_ds = {}
    union_logits = np.full((len(X), N_CLASSES_UNION), -1e9, dtype=np.float32)

    for i, name in enumerate(order):
        m = (ds_ids == i)
        if not np.any(m):
            continue
        logits_local = heads(H[m])[name].cpu().numpy()
        probs_local  = softmax_np(logits_local)
        y_pred_local = probs_local.argmax(1)

        start, width = BLOCKS[name]
        union_logits[m, start:start+width] = logits_local

        if y_u is not None:
            y_true_local = to_local(y_u[m], name)
            acc = accuracy_score(y_true_local, y_pred_local)
            f1m = f1_score(y_true_local, y_pred_local, average="macro")
            per_ds[name] = {"acc": float(acc), "f1_macro": float(f1m)}
        else:
            per_ds[name] = {"acc": float("nan"), "f1_macro": float("nan")}

        if make_plots and y_u is not None and name_prefix:
            if name == "Covertype":
                make_confusion(cfg.save_png, y_true_local, y_pred_local, classes=range(7),
                               title=f"{name_prefix} {MODEL_TAG} Covertype • Confusion")
                make_multiclass_roc(cfg.save_png, y_true_local, probs_local, n_classes=7,
                                    title=f"{name_prefix} {MODEL_TAG} Covertype • ROC")
                make_pred_vs_true_counts(cfg.save_png, y_true_local, y_pred_local, classes=range(7),
                                         title=f"{name_prefix} {MODEL_TAG} Covertype • Pred vs True")
            else:
                pos_prob = probs_local[:, 1]
                make_confusion(cfg.save_png, y_true_local, y_pred_local, classes=[0, 1],
                               title=f"{name_prefix} {MODEL_TAG} {name} • Confusion")
                make_binary_roc(cfg.save_png, y_true_local, pos_prob,
                                title=f"{name_prefix} {MODEL_TAG} {name} • ROC")
                make_pred_vs_true_counts(cfg.save_png, y_true_local, y_pred_local, classes=[0, 1],
                                         title=f"{name_prefix} {MODEL_TAG} {name} • Pred vs True")
                make_binary_prob_by_true(cfg.save_png, y_true_local, pos_prob,
                                         title=f"{name_prefix} {MODEL_TAG} {name} • p(pos) by True")

    u_probs = softmax_np(union_logits)
    y_pred_union = u_probs.argmax(1)
    if y_u is not None:
        union_acc = accuracy_score(y_u, y_pred_union)
        union_f1  = f1_score(y_u, y_pred_union, average="macro")
    else:
        union_acc = float("nan")
        union_f1  = float("nan")

    if make_plots and y_u is not None and name_prefix:
        make_confusion(cfg.save_png, y_u, y_pred_union, classes=range(N_CLASSES_UNION),
                       title=f"{name_prefix} {MODEL_TAG} Unified • Confusion")
        make_multiclass_roc(cfg.save_png, y_u, u_probs, n_classes=N_CLASSES_UNION,
                            title=f"{name_prefix} {MODEL_TAG} Unified • ROC")

    f1s  = [per_ds[k]["f1_macro"] for k in per_ds]
    accs = [per_ds[k]["acc"] for k in per_ds]
    mean_f1  = float(np.nanmean(f1s)) if len(f1s) else float("nan")
    mean_acc = float(np.nanmean(accs)) if len(accs) else float("nan")
    return {
        "union_acc": float(union_acc),
        "union_f1": float(union_f1),
        "mean_acc": mean_acc,
        "mean_f1":  mean_f1,
        "per_ds":   per_ds,
        "y_pred_unified": y_pred_union,
        "proba_union": u_probs,
    }

# =========================
# Train helpers
# =========================
def _amp_helpers(device, use_amp=True):
    if torch.cuda.is_available() and (device == "cuda") and use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        def autocast_ctx(): return torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        class _NoScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
        scaler = _NoScaler()
        def autocast_ctx(): return nullcontext()
    return scaler, autocast_ctx

def make_model_and_opt(hp, input_dim, device, ytr, dstr, order):
    trunk = Trunk(input_dim, hidden=hp["HIDDEN"], dropout=hp["DROPOUT"]).to(device)
    heads = Heads(hidden=hp["HIDDEN"]).to(device)
    w_map = class_weights_per_dataset(ytr, dstr, order)
    loss_cov = nn.CrossEntropyLoss(weight=_safe_weight(w_map, "Covertype", device))
    loss_hig = nn.CrossEntropyLoss(weight=_safe_weight(w_map, "Higgs", device))
    loss_hel = nn.CrossEntropyLoss(weight=_safe_weight(w_map, "HELOC", device))
    opt = torch.optim.Adam(list(trunk.parameters()) + list(heads.parameters()),
                           lr=hp["LR"], weight_decay=hp["WEIGHT_DECAY"])
    return trunk, heads, loss_cov, loss_hig, loss_hel, opt

def train_one_epoch(trunk, heads, criterion, opt, loader, device, order, scaler, autocast_ctx):
    trunk.train(); heads.train()
    loss_cov, loss_hig, loss_hel = criterion
    total_loss = 0.0
    for Xb, yb_u, dsb in loader:
        Xb = Xb.to(device); yb_u = yb_u.to(device); dsb = dsb.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            H = trunk(Xb)
            logit = heads(H)
            L = 0.0; parts = 0
            for i, name in enumerate(order):
                mask = (dsb == i)
                if not torch.any(mask): continue
                y_loc = to_local(yb_u[mask].cpu().numpy(), name)
                y_loc = torch.tensor(y_loc, dtype=torch.long, device=device)
                if name == "Covertype":
                    L += loss_cov(logit[name][mask], y_loc); parts += 1
                elif name == "Higgs":
                    L += loss_hig(logit[name][mask], y_loc); parts += 1
                else:
                    L += loss_hel(logit[name][mask], y_loc); parts += 1
            L = L / max(parts, 1)
        scaler.scale(L).backward()
        if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(list(trunk.parameters()) + list(heads.parameters()),
                                           GRAD_CLIP_NORM)
        scaler.step(opt)
        scaler.update()
        total_loss += float(L.detach().cpu())
    return total_loss / max(1, len(loader))

def train_fold_with_early_stopping(hp, Xtr, ytr, dstr, Xva, yva, dsva, order, device, log_prefix=""):
    train_ds = UnifiedDataset(Xtr, ytr, dstr)
    sampler  = make_balanced_sampler_by_ds_and_class(dstr, ytr, order)
    train_ld = DataLoader(train_ds, batch_size=hp["BATCH_SIZE"], sampler=sampler,
                          drop_last=False, **get_loader_kwargs(device))
    trunk, heads, lc, lh, ll, opt = make_model_and_opt(hp, Xtr.shape[1], device, ytr, dstr, order)
    scaler, autocast_ctx = _amp_helpers(device, USE_AMP)

    best = {"score": -math.inf, "epoch": 0, "state": None, "val": None}
    no_improve = 0
    for epoch in range(1, EPOCHS_MAX + 1):
        _ = train_one_epoch(trunk, heads, (lc, lh, ll), opt, train_ld, device, order, scaler, autocast_ctx)
        # one pass validation
        res = evaluate_multi(SimpleNamespace(save_png=None),
                             order, trunk, heads, Xva, yva, dsva, device, make_plots=False)
        score = res["mean_f1"]

        if PRINT_EPOCH_METRICS and (epoch % EPOCH_LOG_EVERY == 0):
            cov = res["per_ds"].get("Covertype", {"acc": float("nan"), "f1_macro": float("nan")})
            hig = res["per_ds"].get("Higgs",    {"acc": float("nan"), "f1_macro": float("nan")})
            hel = res["per_ds"].get("HELOC",    {"acc": float("nan"), "f1_macro": float("nan")})
            tqdm.write(
                f"{log_prefix} epoch {epoch:03d} | "
                f"val mean_f1={score:.4f} | union_acc={res['union_acc']:.4f} union_f1={res['union_f1']:.4f} | "
                f"cov_f1={cov['f1_macro']:.4f} cov_acc={cov['acc']:.4f} | "
                f"hig_f1={hig['f1_macro']:.4f} hig_acc={hig['acc']:.4f} | "
                f"hel_f1={hel['f1_macro']:.4f} hel_acc={hel['acc']:.4f}"
            )

        if score > best["score"]:
            best["score"] = score
            best["epoch"] = epoch
            best["state"] = {"trunk": trunk.state_dict(), "heads": heads.state_dict(), "config": {**hp}}
            best["val"]   = res
            no_improve = 0
            tqdm.write(f"★ fold new best mean_f1={score:.4f} at epoch {epoch}")
        else:
            no_improve += 1
            if epoch >= max(WARMUP_EPOCHS, 1) and no_improve >= PATIENCE:
                tqdm.write(f"Early stopping at epoch {epoch} (no improve {PATIENCE})")
                break
    return best

def train_fixed_epochs(hp, Xtr, ytr, dstr, order, device, epochs_fixed):
    train_ds = UnifiedDataset(Xtr, ytr, dstr)
    sampler  = make_balanced_sampler_by_ds_and_class(dstr, ytr, order)
    train_ld = DataLoader(train_ds, batch_size=hp["BATCH_SIZE"], sampler=sampler,
                          drop_last=False, **get_loader_kwargs(device))
    trunk, heads, lc, lh, ll, opt = make_model_and_opt(hp, Xtr.shape[1], device, ytr, dstr, order)
    scaler, autocast_ctx = _amp_helpers(device, USE_AMP)
    for _epoch in tqdm(range(1, max(1, epochs_fixed) + 1), leave=False, desc="final_refit_epochs"):
        _ = train_one_epoch(trunk, heads, (lc, lh, ll), opt, train_ld, device, order, scaler, autocast_ctx)
    return trunk, heads

# =========================
# Grid search with K-Fold CV
# =========================
def run_grid_kfold(hp_grid, X, y, ds_ids, order, device, k_folds):
    trials = []
    fold_rows = []
    best_overall = {"score": -math.inf, "trial_idx": -1, "hp": None, "fold_epochs": None}

    strat_labels = make_strat_label(y, ds_ids, order)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    for i, hp in enumerate(hp_grid, start=1):
        fold_scores, fold_epochs = [], []
        t0_trial = time.time()
        for f, (tr_idx, va_idx) in enumerate(skf.split(X, strat_labels), start=1):
            set_seed(SEED + f + i * 37)
            Xtr, ytr, dstr = X[tr_idx], y[tr_idx], ds_ids[tr_idx]
            Xva, yva, dsva = X[va_idx], y[va_idx], ds_ids[va_idx]
            best = train_fold_with_early_stopping(hp, Xtr, ytr, dstr, Xva, yva, dsva, order, device,
                                                  log_prefix=f"[trial {i} fold {f}]")
            fold_scores.append(best["score"])
            fold_epochs.append(best["epoch"])
            fold_rows.append({
                "trial": i, "fold": f,
                "HIDDEN": hp["HIDDEN"], "DROPOUT": hp["DROPOUT"],
                "LR": hp["LR"], "WEIGHT_DECAY": hp["WEIGHT_DECAY"],
                "BATCH_SIZE": hp["BATCH_SIZE"],
                "best_epoch": best["epoch"],
                "mean_f1": best["score"],
                "cov_f1": best["val"]["per_ds"].get("Covertype",{}).get("f1_macro", float("nan")),
                "hig_f1": best["val"]["per_ds"].get("Higgs",{}).get("f1_macro", float("nan")),
                "hel_f1": best["val"]["per_ds"].get("HELOC",{}).get("f1_macro", float("nan")),
                "mean_acc": best["val"]["mean_acc"],
            })

        dt = time.time() - t0_trial
        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        std_score  = float(np.std(fold_scores))  if fold_scores else float("inf")
        med_epoch  = int(np.median(fold_epochs)) if fold_epochs else 1

        trials.append({
            "trial": i,
            "HIDDEN": hp["HIDDEN"], "DROPOUT": hp["DROPOUT"],
            "LR": hp["LR"], "WEIGHT_DECAY": hp["WEIGHT_DECAY"], "BATCH_SIZE": hp["BATCH_SIZE"],
            "k": k_folds,
            "cv_mean_f1": mean_score, "cv_std_f1": std_score, "cv_median_epoch": med_epoch,
            "secs": dt,
        })

        # Selection: higher mean_f1, tie-break lower std, then faster
        is_better = (mean_score > best_overall["score"]) or (
            math.isclose(mean_score, best_overall["score"], rel_tol=1e-6) and
            (std_score < best_overall.get("cv_std_f1", float("inf")) or
             (math.isclose(std_score, best_overall.get("cv_std_f1", float("inf")), rel_tol=1e-6) and dt < best_overall.get("secs", float("inf"))))
        )
        if is_better:
            best_overall.update(score=mean_score, trial_idx=i-1, hp=hp, fold_epochs=fold_epochs,
                                cv_std_f1=std_score, secs=dt)
            tqdm.write(f"★ new best trial {i}: cv_mean_f1={mean_score:.4f} (±{std_score:.4f}), median_epoch={med_epoch}")

    trials_df = pd.DataFrame(trials)
    folds_df  = pd.DataFrame(fold_rows)
    return best_overall, trials_df, folds_df

# =========================
# Main
# =========================
def main():
    print("\n" + "="*80 + "\nENVIRONMENT\n" + "="*80, flush=True)
    print(f"[Python] {platform.python_version()}", flush=True)
    print(f"[Torch]  {torch.__version__}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Using device] {device}", flush=True)

    set_seed()
    PLOTS.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    save_png = make_saver(PLOTS, tag="v7_multihead")

    # Data
    Z, y, ds_ids, order = load_trainval_latents_and_meta()

    # Stratified TEST hold-out; TrainVal used for CV
    idx_trainval, idx_test = stratified_test_holdout(ds_ids, y, order, TEST_RATIO, SEED)
    Xtv, ytv, dstv = Z[idx_trainval], y[idx_trainval], ds_ids[idx_trainval]
    Xte, yte, dste = Z[idx_test],       y[idx_test],   ds_ids[idx_test]

    # Auto-calibrate K
    k_folds = auto_adjust_kfold(idx_trainval, y, ds_ids, order, requested_k=K_FOLDS)
    print(f"[CV] Using K={k_folds} folds.", flush=True)

    # Grid search via K-Fold CV
    best_overall, trials_df, folds_df = run_grid_kfold(HP_GRID, Xtv, ytv, dstv, order, device, k_folds)
    trials_df.to_csv(CSV_TRIALS_LOG, index=False)
    folds_df.to_csv(CSV_TRIALS_FOLDS, index=False)
    print(f"Saved trials log to {CSV_TRIALS_LOG}")
    print(f"Saved per-fold log to {CSV_TRIALS_FOLDS}")

    # Best HP & fixed-epoch refit on ALL TrainVal (no validation = no leakage)
    hp = best_overall["hp"] if best_overall["hp"] is not None else BASE_HP
    best_epoch_median = int(np.median(best_overall["fold_epochs"])) if best_overall["fold_epochs"] else 1
    print(f"[REFIT] Best HP: {hp} | fixed_epochs={best_epoch_median}", flush=True)
    set_seed(SEED + 999)
    trunk, heads = train_fixed_epochs(hp, Xtv, ytv, dstv, order, device, epochs_fixed=max(1, best_epoch_median))

    # Save checkpoint and config
    torch.save({"trunk": trunk.state_dict(), "heads": heads.state_dict(), "config": {**hp}, "epochs_refit": best_epoch_median}, CKPT)
    with open(TXT_BEST, "w") as f:
        f.write("Best HP configuration (multihead v7)\n")
        f.write(json.dumps(hp, indent=2))
        f.write(f"\nMedian best epoch across folds: {best_epoch_median}\n")
        f.write(f"CV mean_f1: {best_overall['score']:.6f} ± {best_overall.get('cv_std_f1', float('nan')):.6f}\n")
    print(f"Saved best checkpoint to {CKPT}")
    print(f"Wrote best config to {TXT_BEST}")

    # Validation-on-refit plots ONCE (tiny slice from TrainVal; visualization only)
    if PLOT_AFTER_FINAL_ONLY:
        _, idx_vis_val = stratified_test_holdout(dstv, ytv, order, test_ratio=0.10, seed=SEED+123)
        evaluate_multi(SimpleNamespace(save_png=save_png),
                       order, trunk, heads, Xtv[idx_vis_val], ytv[idx_vis_val], dstv[idx_vis_val], device,
                       make_plots=True, name_prefix="POST-REFIT (TrainVal slice)")

    # Test with plots ONCE
    print("\n" + "="*80 + "\nTEST EVALUATION\n" + "="*80, flush=True)
    res_te = evaluate_multi(SimpleNamespace(save_png=make_saver(PLOTS, tag="v7_multihead")),
                            order, trunk, heads, Xte, yte, dste, device,
                            make_plots=True, name_prefix="TEST")

    # Save test CSVs
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

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
