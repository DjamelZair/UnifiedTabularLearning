# scripts/tabster_viz.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

PALETTE = {
    "covtype":   "#78FECF",
    "higgs":     "#E5EAFA",
    "heloc":     "#F1D302",
    "true":      "#155965",
    "pred":      "#BDC7EA",
    "roc":       "#155965",
    "ref":       "#888888",
}

def configure_matplotlib():
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.figsize": (14, 9),
        "axes.titlesize": 24,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.facecolor": "#ffffff",
        "axes.titlepad": 12,
    })

def make_saver(plot_dir: Path, tag: str):
    plot_dir.mkdir(parents=True, exist_ok=True)
    def save_png(fig, name):
        fig.tight_layout()
        p = plot_dir / f"{name}_{tag}.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"Saved {p}", flush=True)
    return save_png

def plot_dataset_size_imbalance(save_png, ds_ids, order, prefix="dataset_size_imbalance"):
    names  = list(order)
    counts = [int((ds_ids == i).sum()) for i in range(len(order))]
    total  = sum(counts)
    colors = [PALETTE["covtype"], PALETTE["higgs"], PALETTE["heloc"]]
    fig, ax = plt.subplots()
    ax.grid(True, axis="y", alpha=0.3)
    bars = ax.bar(names, counts, color=colors, edgecolor="#222222", linewidth=1.2)
    ax.set_title("Dataset Size Imbalance: Counts and Percent")
    ax.set_ylabel("Row Count")
    labels = [f"{v:,}\n({(100.0*v/max(total,1)):.1f}%)" for v in counts]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=12)
    save_png(fig, prefix)

def make_confusion(save_png, y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    fig, ax = plt.subplots()
    ax.grid(False)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels([str(c) for c in classes])
    ax.set_yticks(np.arange(len(classes))); ax.set_yticklabels([str(c) for c in classes])
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=11, color=("white" if cm[i, j] > thresh else "black"))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.ax.grid(False)
    save_png(fig, title.lower().replace(" ", "_"))

def make_multiclass_roc(save_png, y_true, proba, n_classes, title):
    Y = np.eye(n_classes)[y_true]
    fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    grid = np.linspace(0, 1, 400)
    tprs = []
    for c in range(n_classes):
        f, t, _ = roc_curve((y_true == c).astype(int), proba[:, c])
        t_interp = np.interp(grid, f, t); t_interp[0] = 0.0
        tprs.append(t_interp)
    mean_tpr = np.mean(np.stack(tprs, axis=0), axis=0)
    auc_macro = auc(grid, mean_tpr)
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)
    ax.plot(grid, mean_tpr, lw=3.0, label=f"Macro AUC = {auc_macro:.3f}")
    ax.plot(fpr_micro, tpr_micro, lw=3.0, color="#222222", label=f"Micro AUC = {auc_micro:.3f}")
    ax.plot([0, 1], [0, 1], lw=2.0, color=PALETTE["ref"])
    ax.set_title(title); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    save_png(fig, title.lower().replace(" ", "_"))

def make_binary_roc(save_png, y_true_local, pos_prob, title):
    fpr, tpr, _ = roc_curve(y_true_local, pos_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)
    ax.plot(fpr, tpr, lw=3.0, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], lw=2.0, color=PALETTE["ref"])
    ax.set_title(title); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend()
    save_png(fig, title.lower().replace(" ", "_"))

def make_pred_vs_true_counts(save_png, y_true, y_pred, classes, title):
    true_counts = np.bincount(y_true, minlength=len(classes))
    pred_counts = np.bincount(y_pred, minlength=len(classes))
    idx = np.arange(len(classes)); width = 0.42
    fig, ax = plt.subplots()
    ax.grid(True, axis="y", alpha=0.3)
    bars1 = ax.bar(idx - width/2, true_counts, width, label="True", color=PALETTE["true"])
    bars2 = ax.bar(idx + width/2, pred_counts, width, label="Predicted", color=PALETTE["pred"])
    ax.bar_label(bars1, labels=[f"{int(v):,}" for v in true_counts], padding=3, fontsize=10)
    ax.bar_label(bars2, labels=[f"{int(v):,}" for v in pred_counts], padding=3, fontsize=10)
    ax.set_title(title); ax.set_xlabel("Class"); ax.set_ylabel("Count")
    ax.set_xticks(idx); ax.set_xticklabels([str(c) for c in classes])
    ax.legend()
    save_png(fig, title.lower().replace(" ", "_"))

def make_binary_prob_by_true(save_png, y_true_local, pos_prob, title):
    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots()
    ax.grid(True, axis="y", alpha=0.3)
    ax.hist(pos_prob[y_true_local == 1], bins=bins, alpha=0.55, label="True positive", color=PALETTE["heloc"])
    ax.hist(pos_prob[y_true_local == 0], bins=bins, alpha=0.40, label="True negative", color=PALETTE["higgs"])
    ax.axvline(0.5, color="#333333", linewidth=2)
    ax.set_title(title); ax.set_xlabel("Predicted probability for positive class"); ax.set_ylabel("Count")
    ax.legend()
    save_png(fig, title.lower().replace(" ", "_"))

# -------- Label-free test plots (no y_true needed) --------

def plot_pred_counts_union(y_pred_unified, save_png, name="test_pred_counts_union"):
    # Show distribution over 11 unified classes
    K = int(np.max(y_pred_unified)) + 1
    counts = np.bincount(y_pred_unified, minlength=K)
    idx = np.arange(K)
    fig, ax = plt.subplots()
    ax.grid(True, axis="y", alpha=0.3)
    bars = ax.bar(idx, counts, color=PALETTE["pred"], edgecolor="#222")
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in counts], padding=3)
    ax.set_title("Unified predictions: class counts")
    ax.set_xlabel("Unified class id"); ax.set_ylabel("Count")
    save_png(fig, name)

def plot_pred_counts_by_dataset(y_pred_unified, ds_ids, order, save_png, name="test_pred_counts_by_dataset"):
    # For readability, only count how many rows per dataset; it doesn’t need labels
    names  = list(order)
    counts = [int((ds_ids == i).sum()) for i in range(len(order))]
    colors = [PALETTE["covtype"], PALETTE["higgs"], PALETTE["heloc"]]
    fig, ax = plt.subplots()
    ax.grid(True, axis="y", alpha=0.3)
    bars = ax.bar(names, counts, color=colors, edgecolor="#222")
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts], padding=4)
    ax.set_title("Test rows per dataset (predictions stacked in this mix)")
    ax.set_ylabel("Row count")
    save_png(fig, name)

def plot_confidence_hist(top_conf, save_png, name="test_confidence_hist"):
    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots()
    ax.grid(True, axis="y", alpha=0.3)
    ax.hist(top_conf, bins=bins, alpha=0.7, color=PALETTE["pred"])
    ax.set_title("Prediction confidence (max softmax)")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    save_png(fig, name)
