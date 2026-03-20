"""
Diffusion-TS Results Visualization
===================================
Generates presentation-ready figures comparing generated vs real time series,
with a framework for side-by-side comparison against PaD-TS.

Usage:
    # After training + sampling, generate all figures for a dataset:
    python visualize_results.py --dataset stocks

    # Compare Diffusion-TS vs PaD-TS (provide PaD-TS .npy path):
    python visualize_results.py --dataset stocks --padts_path /path/to/padts_fake_stocks.npy

    # All datasets at once:
    python visualize_results.py --dataset all

    # Skip metric computation (faster, plots only):
    python visualize_results.py --dataset stocks --skip_metrics
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "real": "#2c3e50",
    "diffts": "#e74c3c",
    "padts": "#3498db",
    "real_fill": "#bdc3c7",
    "diffts_fill": "#e74c3c",
    "padts_fill": "#3498db",
}

# ── Dataset registry ─────────────────────────────────────────────────────────

DATASETS = {
    "stocks": {"features": 6, "seq_len": 24, "feature_names": None},
    "etth":   {"features": 7, "seq_len": 24, "feature_names": None},
    "energy": {"features": 28, "seq_len": 24, "feature_names": None},
    "fmri":   {"features": 50, "seq_len": 24, "feature_names": None},
    "sines":  {"features": 5, "seq_len": 24, "feature_names": None},
    "mujoco": {"features": 14, "seq_len": 24, "feature_names": None},
}


# ── Data loading ─────────────────────────────────────────────────────────────

def find_npy(name, pattern_candidates, base_dirs):
    """Search for a .npy file across candidate patterns and directories."""
    for d in base_dirs:
        for pat in pattern_candidates:
            p = Path(d) / pat.format(name=name)
            if p.exists():
                return p
    return None


def load_dataset(dataset_name, output_dir="OUTPUT", padts_path=None):
    """Load real, Diffusion-TS synthetic, and optionally PaD-TS data."""
    base_dirs = [
        output_dir,
        os.path.join(output_dir, dataset_name),
        os.path.join(output_dir, "samples"),
        ".",
    ]

    seq_len = DATASETS[dataset_name]["seq_len"]

    # Real data (try norm first, then ground truth)
    real_candidates = [
        "{name}_norm_truth_" + str(seq_len) + "_train.npy",
        "{name}_ground_truth_" + str(seq_len) + "_train.npy",
    ]
    real_path = find_npy(dataset_name, real_candidates, base_dirs)
    if real_path is None:
        # Also check for name variants (e.g. "stock" vs "stocks")
        alt_name = dataset_name.rstrip("s") if dataset_name.endswith("s") else dataset_name
        real_path = find_npy(alt_name, real_candidates, base_dirs)

    # Diffusion-TS synthetic data
    fake_candidates = ["ddpm_fake_{name}.npy"]
    fake_path = find_npy(dataset_name, fake_candidates, base_dirs)

    data = {}
    if real_path:
        data["real"] = np.load(str(real_path))
        print(f"  Real data: {real_path}  shape={data['real'].shape}")
    else:
        print(f"  WARNING: No real data found for '{dataset_name}'")

    if fake_path:
        data["diffts"] = np.load(str(fake_path))
        print(f"  Diffusion-TS: {fake_path}  shape={data['diffts'].shape}")
    else:
        print(f"  WARNING: No Diffusion-TS samples found for '{dataset_name}'")

    if padts_path and os.path.exists(padts_path):
        data["padts"] = np.load(padts_path)
        print(f"  PaD-TS: {padts_path}  shape={data['padts'].shape}")

    return data


# ── Metric computation ───────────────────────────────────────────────────────

def compute_context_fid(real, fake, n_iters=3):
    """Compute Context-FID score (requires Utils on path)."""
    try:
        from Utils.context_fid import Context_FID
        scores = []
        for i in range(n_iters):
            idx_r = np.random.choice(len(real), min(1000, len(real)), replace=False)
            idx_f = np.random.choice(len(fake), min(1000, len(fake)), replace=False)
            score = Context_FID(real[idx_r], fake[idx_f])
            scores.append(score)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception as e:
        print(f"    Context-FID failed: {e}")
        return None, None


def compute_cross_corr(real, fake, n_iters=3):
    """Compute cross-correlation score."""
    try:
        from Utils.cross_correlation import CrossCorrelLoss
        loss_fn = CrossCorrelLoss()
        import torch
        scores = []
        for _ in range(n_iters):
            idx_r = np.random.choice(len(real), min(1000, len(real)), replace=False)
            idx_f = np.random.choice(len(fake), min(1000, len(fake)), replace=False)
            r_t = torch.tensor(real[idx_r], dtype=torch.float32)
            f_t = torch.tensor(fake[idx_f], dtype=torch.float32)
            score = loss_fn(r_t, f_t).item()
            scores.append(score)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception as e:
        print(f"    Cross-correlation failed: {e}")
        return None, None


def compute_discriminative(real, fake):
    """Compute discriminative score."""
    try:
        from Utils.discriminative_metric import discriminative_score_metrics
        scores = []
        for _ in range(3):
            idx_r = np.random.choice(len(real), min(500, len(real)), replace=False)
            idx_f = np.random.choice(len(fake), min(500, len(fake)), replace=False)
            score = discriminative_score_metrics(real[idx_r], fake[idx_f])
            if isinstance(score, (list, tuple)):
                score = score[0]
            scores.append(score)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception as e:
        print(f"    Discriminative failed: {e}")
        return None, None


def compute_predictive(real, fake):
    """Compute predictive score."""
    try:
        from Utils.predictive_metric import predictive_score_metrics
        scores = []
        for _ in range(3):
            idx_r = np.random.choice(len(real), min(500, len(real)), replace=False)
            idx_f = np.random.choice(len(fake), min(500, len(fake)), replace=False)
            score = predictive_score_metrics(real[idx_r], fake[idx_f])
            if isinstance(score, (list, tuple)):
                score = score[0]
            scores.append(score)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception as e:
        print(f"    Predictive failed: {e}")
        return None, None


def compute_all_metrics(real, fake, label=""):
    """Compute all four metrics, return dict."""
    print(f"  Computing metrics for {label}...")
    metrics = {}

    m, s = compute_context_fid(real, fake)
    metrics["Context-FID"] = (m, s)
    if m is not None:
        print(f"    Context-FID: {m:.4f} +/- {s:.4f}")

    m, s = compute_cross_corr(real, fake)
    metrics["Cross-Corr"] = (m, s)
    if m is not None:
        print(f"    Cross-Corr:  {m:.4f} +/- {s:.4f}")

    m, s = compute_discriminative(real, fake)
    metrics["Discriminative"] = (m, s)
    if m is not None:
        print(f"    Discriminative: {m:.4f} +/- {s:.4f}")

    m, s = compute_predictive(real, fake)
    metrics["Predictive"] = (m, s)
    if m is not None:
        print(f"    Predictive: {m:.4f} +/- {s:.4f}")

    return metrics


# ── Figure 1: Sample time series comparison ──────────────────────────────────

def plot_sample_series(data, dataset_name, out_dir, n_samples=3, n_features=4):
    """Plot example real vs generated time series side by side."""
    has_padts = "padts" in data
    n_cols = 3 if has_padts else 2
    n_features = min(n_features, data["real"].shape[2])

    fig, axes = plt.subplots(n_features, n_cols, figsize=(4.5 * n_cols, 2.5 * n_features))
    if n_features == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Real", "Diffusion-TS"]
    col_keys = ["real", "diffts"]
    col_colors = [COLORS["real"], COLORS["diffts"]]
    if has_padts:
        col_labels.append("PaD-TS")
        col_keys.append("padts")
        col_colors.append(COLORS["padts"])

    for col_idx, (key, label, color) in enumerate(zip(col_keys, col_labels, col_colors)):
        if key not in data:
            continue
        arr = data[key]
        idxs = np.random.choice(len(arr), n_samples, replace=False)
        for feat in range(n_features):
            ax = axes[feat, col_idx]
            for i, idx in enumerate(idxs):
                alpha = 0.9 - 0.2 * i
                ax.plot(arr[idx, :, feat], color=color, alpha=alpha, linewidth=1.2)
            if feat == 0:
                ax.set_title(label, fontweight="bold", color=color)
            if col_idx == 0:
                ax.set_ylabel(f"Feature {feat}")
            ax.xaxis.set_major_locator(MaxNLocator(5))

    fig.suptitle(f"{dataset_name.upper()} — Sample Time Series", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_samples.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: t-SNE embedding ────────────────────────────────────────────────

def plot_tsne(data, dataset_name, out_dir, max_n=2000):
    """t-SNE of real vs synthetic distributions."""
    has_padts = "padts" in data
    n_cols = 2 if has_padts else 1

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    real = data["real"][:max_n]
    # Flatten: (N, seq, feat) -> (N, seq*feat)
    real_flat = real.reshape(len(real), -1)

    comparisons = [("diffts", "Diffusion-TS", COLORS["diffts"])]
    if has_padts:
        comparisons.append(("padts", "PaD-TS", COLORS["padts"]))

    for ax, (key, label, color) in zip(axes, comparisons):
        fake = data[key][:max_n]
        fake_flat = fake.reshape(len(fake), -1)

        combined = np.concatenate([real_flat, fake_flat], axis=0)
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
        embedded = tsne.fit_transform(combined)

        n_real = len(real_flat)
        ax.scatter(embedded[:n_real, 0], embedded[:n_real, 1],
                   c=COLORS["real"], s=8, alpha=0.5, label="Real", edgecolors="none")
        ax.scatter(embedded[n_real:, 0], embedded[n_real:, 1],
                   c=color, s=8, alpha=0.5, label=label, edgecolors="none")
        ax.legend(markerscale=3, framealpha=0.8)
        ax.set_title(f"Real vs {label}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{dataset_name.upper()} — t-SNE", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_tsne.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: PCA embedding ──────────────────────────────────────────────────

def plot_pca(data, dataset_name, out_dir, max_n=2000):
    """PCA projection of real vs synthetic."""
    has_padts = "padts" in data
    n_cols = 2 if has_padts else 1

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    real = data["real"][:max_n]
    real_flat = real.reshape(len(real), -1)

    pca = PCA(n_components=2)
    pca.fit(real_flat)

    real_pca = pca.transform(real_flat)

    comparisons = [("diffts", "Diffusion-TS", COLORS["diffts"])]
    if has_padts:
        comparisons.append(("padts", "PaD-TS", COLORS["padts"]))

    for ax, (key, label, color) in zip(axes, comparisons):
        fake = data[key][:max_n]
        fake_flat = fake.reshape(len(fake), -1)
        fake_pca = pca.transform(fake_flat)

        ax.scatter(real_pca[:, 0], real_pca[:, 1],
                   c=COLORS["real"], s=8, alpha=0.4, label="Real", edgecolors="none")
        ax.scatter(fake_pca[:, 0], fake_pca[:, 1],
                   c=color, s=8, alpha=0.4, label=label, edgecolors="none")
        ax.legend(markerscale=3, framealpha=0.8)
        ax.set_title(f"Real vs {label}", fontweight="bold")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    fig.suptitle(f"{dataset_name.upper()} — PCA", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_pca.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Marginal distributions (KDE) ───────────────────────────────────

def plot_marginals(data, dataset_name, out_dir, n_features=6):
    """KDE of per-feature marginal distributions."""
    n_features = min(n_features, data["real"].shape[2])
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(n_features):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        real_feat = data["real"][:, :, i].flatten()
        ax.hist(real_feat, bins=60, density=True, alpha=0.35,
                color=COLORS["real"], label="Real")

        diffts_feat = data["diffts"][:, :, i].flatten()
        ax.hist(diffts_feat, bins=60, density=True, alpha=0.35,
                color=COLORS["diffts"], label="Diffusion-TS", histtype="step", linewidth=1.5)

        if "padts" in data:
            padts_feat = data["padts"][:, :, i].flatten()
            ax.hist(padts_feat, bins=60, density=True, alpha=0.35,
                    color=COLORS["padts"], label="PaD-TS", histtype="step", linewidth=1.5)

        ax.set_title(f"Feature {i}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_features, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"{dataset_name.upper()} — Marginal Distributions",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_marginals.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Temporal autocorrelation ────────────────────────────────────────

def plot_autocorrelation(data, dataset_name, out_dir, max_lag=12, n_features=4):
    """Compare autocorrelation structure of real vs generated."""
    n_features = min(n_features, data["real"].shape[2])

    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 3.5))
    if n_features == 1:
        axes = [axes]

    lags = np.arange(1, max_lag + 1)

    for feat, ax in zip(range(n_features), axes):
        for key, label, color in [("real", "Real", COLORS["real"]),
                                   ("diffts", "Diffusion-TS", COLORS["diffts"]),
                                   ("padts", "PaD-TS", COLORS["padts"])]:
            if key not in data:
                continue
            series = data[key][:, :, feat]  # (N, T)
            acorrs = []
            for lag in lags:
                x = series[:, :-lag]
                y = series[:, lag:]
                # Mean correlation across samples
                corr = np.mean([np.corrcoef(x[i], y[i])[0, 1]
                                for i in range(min(500, len(series)))
                                if np.std(x[i]) > 1e-8 and np.std(y[i]) > 1e-8])
                acorrs.append(corr)
            ax.plot(lags, acorrs, marker="o", markersize=3, color=color,
                    label=label, linewidth=1.5)

        ax.set_title(f"Feature {feat}", fontsize=10)
        ax.set_xlabel("Lag")
        if feat == 0:
            ax.set_ylabel("Autocorrelation")
            ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(f"{dataset_name.upper()} — Autocorrelation",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset_name}_autocorr.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Metric comparison bar chart (across datasets) ──────────────────

def plot_metric_comparison(all_metrics, out_dir):
    """Bar chart comparing Diffusion-TS (and PaD-TS) across datasets and metrics."""
    metric_names = ["Context-FID", "Cross-Corr", "Discriminative", "Predictive"]
    datasets_with_metrics = [d for d in all_metrics if all_metrics[d]]

    if not datasets_with_metrics:
        print("  No metrics to plot.")
        return

    has_padts = any("padts" in all_metrics[d] for d in datasets_with_metrics)
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))

    for ax, metric in zip(axes, metric_names):
        ds_names = []
        diffts_vals, diffts_errs = [], []
        padts_vals, padts_errs = [], []

        for ds in datasets_with_metrics:
            if "diffts" in all_metrics[ds] and metric in all_metrics[ds]["diffts"]:
                m, s = all_metrics[ds]["diffts"][metric]
                if m is not None:
                    ds_names.append(ds)
                    diffts_vals.append(m)
                    diffts_errs.append(s or 0)

                    if has_padts and "padts" in all_metrics[ds] and metric in all_metrics[ds]["padts"]:
                        pm, ps = all_metrics[ds]["padts"][metric]
                        padts_vals.append(pm if pm is not None else 0)
                        padts_errs.append(ps if ps is not None else 0)
                    else:
                        padts_vals.append(0)
                        padts_errs.append(0)

        if not ds_names:
            ax.set_visible(False)
            continue

        x = np.arange(len(ds_names))
        width = 0.35 if has_padts else 0.5
        offset = width / 2 if has_padts else 0

        ax.bar(x - offset, diffts_vals, width, yerr=diffts_errs,
               color=COLORS["diffts"], alpha=0.8, label="Diffusion-TS", capsize=3)
        if has_padts and any(v > 0 for v in padts_vals):
            ax.bar(x + offset, padts_vals, width, yerr=padts_errs,
                   color=COLORS["padts"], alpha=0.8, label="PaD-TS", capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Quantitative Metrics Comparison (lower is better)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "metrics_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 7: Summary dashboard per dataset ──────────────────────────────────

def plot_dashboard(data, metrics, dataset_name, out_dir):
    """Single-page dashboard: samples + t-SNE + PCA + marginals + metrics."""
    has_padts = "padts" in data
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Top-left: sample series (2 cols)
    ax_samples = fig.add_subplot(gs[0, 0:2])
    n_show = min(3, data["real"].shape[2])
    for feat in range(n_show):
        idx = np.random.randint(len(data["real"]))
        ax_samples.plot(data["real"][idx, :, feat], color=COLORS["real"],
                        alpha=0.7, linewidth=1.2, label="Real" if feat == 0 else None)
        if "diffts" in data:
            idx_f = np.random.randint(len(data["diffts"]))
            ax_samples.plot(data["diffts"][idx_f, :, feat], color=COLORS["diffts"],
                            alpha=0.7, linewidth=1.2, linestyle="--",
                            label="Diffusion-TS" if feat == 0 else None)
    ax_samples.legend(fontsize=8)
    ax_samples.set_title("Sample Series", fontweight="bold")

    # Top-right: t-SNE (2 cols)
    ax_tsne = fig.add_subplot(gs[0, 2:4])
    max_n = min(1500, len(data["real"]))
    real_flat = data["real"][:max_n].reshape(max_n, -1)
    if "diffts" in data:
        fake_flat = data["diffts"][:max_n].reshape(min(max_n, len(data["diffts"])), -1)
        combined = np.concatenate([real_flat, fake_flat], axis=0)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        emb = tsne.fit_transform(combined)
        n_r = len(real_flat)
        ax_tsne.scatter(emb[:n_r, 0], emb[:n_r, 1], c=COLORS["real"], s=6,
                        alpha=0.4, label="Real", edgecolors="none")
        ax_tsne.scatter(emb[n_r:, 0], emb[n_r:, 1], c=COLORS["diffts"], s=6,
                        alpha=0.4, label="Diffusion-TS", edgecolors="none")
        ax_tsne.legend(markerscale=3, fontsize=8)
    ax_tsne.set_title("t-SNE", fontweight="bold")
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])

    # Bottom-left: marginals (2 cols)
    ax_marg = fig.add_subplot(gs[1, 0:2])
    feat_idx = 0
    real_feat = data["real"][:, :, feat_idx].flatten()
    ax_marg.hist(real_feat, bins=50, density=True, alpha=0.4,
                 color=COLORS["real"], label="Real")
    if "diffts" in data:
        fake_feat = data["diffts"][:, :, feat_idx].flatten()
        ax_marg.hist(fake_feat, bins=50, density=True, alpha=0.4,
                     color=COLORS["diffts"], label="Diffusion-TS")
    if "padts" in data:
        padts_feat = data["padts"][:, :, feat_idx].flatten()
        ax_marg.hist(padts_feat, bins=50, density=True, alpha=0.4,
                     color=COLORS["padts"], label="PaD-TS")
    ax_marg.legend(fontsize=8)
    ax_marg.set_title("Marginal Distribution (Feature 0)", fontweight="bold")

    # Bottom-right: metrics table (2 cols)
    ax_table = fig.add_subplot(gs[1, 2:4])
    ax_table.axis("off")
    if metrics:
        rows = []
        for metric_name in ["Context-FID", "Cross-Corr", "Discriminative", "Predictive"]:
            row = [metric_name]
            for method in ["diffts", "padts"]:
                if method in metrics and metric_name in metrics[method]:
                    m, s = metrics[method][metric_name]
                    if m is not None:
                        row.append(f"{m:.4f} +/- {s:.4f}")
                    else:
                        row.append("--")
                else:
                    row.append("--")
            rows.append(row)

        col_labels = ["Metric", "Diffusion-TS"]
        if has_padts:
            col_labels.append("PaD-TS")
        else:
            rows = [r[:2] for r in rows]

        table = ax_table.table(cellText=rows, colLabels=col_labels,
                                loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        ax_table.set_title("Quantitative Metrics (lower is better)", fontweight="bold")
    else:
        ax_table.text(0.5, 0.5, "Metrics not computed\n(use --skip_metrics=False)",
                      ha="center", va="center", fontsize=12, color="gray")

    fig.suptitle(f"{dataset_name.upper()} — Results Dashboard",
                 fontsize=18, fontweight="bold", y=1.01)
    path = os.path.join(out_dir, f"{dataset_name}_dashboard.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def process_dataset(dataset_name, args):
    """Generate all figures for one dataset."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()}")
    print(f"{'='*60}")

    data = load_dataset(dataset_name, output_dir=args.output_dir, padts_path=args.padts_path)

    if "real" not in data:
        print(f"  Skipping {dataset_name}: no real data found.")
        return {}

    if "diffts" not in data:
        print(f"  Skipping {dataset_name}: no Diffusion-TS samples found.")
        print(f"  (Run sampling first: python main.py --name {dataset_name} "
              f"--config_file Config/{dataset_name}.yaml --gpu 0 --sample 0 --milestone N)")
        return {}

    out_dir = os.path.join(args.fig_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Plots (always)
    np.random.seed(42)
    plot_sample_series(data, dataset_name, out_dir)
    plot_tsne(data, dataset_name, out_dir)
    plot_pca(data, dataset_name, out_dir)
    plot_marginals(data, dataset_name, out_dir)
    plot_autocorrelation(data, dataset_name, out_dir)

    # Metrics (optional)
    metrics = {}
    if not args.skip_metrics:
        metrics["diffts"] = compute_all_metrics(data["real"], data["diffts"], "Diffusion-TS")
        if "padts" in data:
            metrics["padts"] = compute_all_metrics(data["real"], data["padts"], "PaD-TS")

    # Dashboard
    plot_dashboard(data, metrics, dataset_name, out_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Diffusion-TS visualization")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all' (default: all)")
    parser.add_argument("--output_dir", type=str, default="OUTPUT",
                        help="Directory containing .npy outputs (default: OUTPUT)")
    parser.add_argument("--fig_dir", type=str, default="figures/results",
                        help="Directory to save figures (default: figures/results)")
    parser.add_argument("--padts_path", type=str, default=None,
                        help="Path to PaD-TS .npy file for comparison")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip metric computation (faster)")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    if args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]

    all_metrics = {}
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}")
            continue
        all_metrics[ds] = process_dataset(ds, args)

    # Cross-dataset metric comparison chart
    if not args.skip_metrics and len(datasets) > 1:
        print(f"\n{'='*60}")
        print("  Cross-Dataset Metric Comparison")
        print(f"{'='*60}")
        plot_metric_comparison(all_metrics, args.fig_dir)

    print(f"\nDone! Figures saved to: {args.fig_dir}/")


if __name__ == "__main__":
    main()
