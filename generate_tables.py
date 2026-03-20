"""
Diffusion-TS Results Tables
============================
Generates publication-quality LaTeX and CSV tables of quantitative metrics,
with Diffusion-TS vs PaD-TS comparison support.

Usage:
    # Compute metrics and generate tables for all datasets:
    python generate_tables.py --dataset all

    # Single dataset:
    python generate_tables.py --dataset stocks

    # Include PaD-TS comparison:
    python generate_tables.py --dataset stocks --padts_path /path/to/padts_fake_stocks.npy

    # Load pre-computed metrics from JSON instead of recomputing:
    python generate_tables.py --load_metrics results/metrics.json

    # Save computed metrics to JSON for later use:
    python generate_tables.py --dataset all --save_metrics results/metrics.json
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Registry ─────────────────────────────────────────────────────────────────

DATASETS = {
    "stocks": {"features": 6, "seq_len": 24},
    "etth":   {"features": 7, "seq_len": 24},
    "energy": {"features": 28, "seq_len": 24},
    "fmri":   {"features": 50, "seq_len": 24},
    "sines":  {"features": 5, "seq_len": 24},
    "mujoco": {"features": 14, "seq_len": 24},
}

DATASET_DISPLAY = {
    "stocks": "Stocks",
    "etth": "ETTh1",
    "energy": "Energy",
    "fmri": "fMRI",
    "sines": "Sine",
    "mujoco": "MuJoCo",
}

METRIC_NAMES = ["Context FID", "Cross Corr.", "Discriminative", "Predictive"]

METRIC_LATEX = {
    "Context FID": r"Context FID $\downarrow$",
    "Cross Corr.": r"Cross Corr.\ $\downarrow$",
    "Discriminative": r"Discriminative $\downarrow$",
    "Predictive": r"Predictive $\downarrow$",
}


# ── Data loading (reuse from visualize_results) ─────────────────────────────

def find_npy(name, pattern_candidates, base_dirs):
    for d in base_dirs:
        for pat in pattern_candidates:
            p = Path(d) / pat.format(name=name)
            if p.exists():
                return p
    return None


def load_dataset(dataset_name, output_dir="OUTPUT", padts_path=None):
    base_dirs = [
        output_dir,
        os.path.join(output_dir, dataset_name),
        os.path.join(output_dir, "samples"),
        ".",
    ]

    seq_len = DATASETS[dataset_name]["seq_len"]

    real_candidates = [
        "{name}_norm_truth_" + str(seq_len) + "_train.npy",
        "{name}_ground_truth_" + str(seq_len) + "_train.npy",
    ]
    real_path = find_npy(dataset_name, real_candidates, base_dirs)
    if real_path is None:
        alt_name = dataset_name.rstrip("s") if dataset_name.endswith("s") else dataset_name
        real_path = find_npy(alt_name, real_candidates, base_dirs)

    fake_candidates = ["ddpm_fake_{name}.npy"]
    fake_path = find_npy(dataset_name, fake_candidates, base_dirs)

    data = {}
    if real_path:
        data["real"] = np.load(str(real_path))
    if fake_path:
        data["diffts"] = np.load(str(fake_path))
    if padts_path and os.path.exists(padts_path):
        data["padts"] = np.load(padts_path)

    return data


# ── Metric computation ───────────────────────────────────────────────────────

def compute_context_fid(real, fake, n_iters=3):
    try:
        from Utils.context_fid import Context_FID
        scores = []
        for _ in range(n_iters):
            idx_r = np.random.choice(len(real), min(1000, len(real)), replace=False)
            idx_f = np.random.choice(len(fake), min(1000, len(fake)), replace=False)
            scores.append(Context_FID(real[idx_r], fake[idx_f]))
        return float(np.mean(scores)), float(np.std(scores))
    except Exception as e:
        print(f"    Context FID failed: {e}")
        return None, None


def compute_cross_corr(real, fake, n_iters=3):
    try:
        from Utils.cross_correlation import CrossCorrelLoss
        import torch
        loss_fn = CrossCorrelLoss()
        scores = []
        for _ in range(n_iters):
            idx_r = np.random.choice(len(real), min(1000, len(real)), replace=False)
            idx_f = np.random.choice(len(fake), min(1000, len(fake)), replace=False)
            r_t = torch.tensor(real[idx_r], dtype=torch.float32)
            f_t = torch.tensor(fake[idx_f], dtype=torch.float32)
            scores.append(loss_fn(r_t, f_t).item())
        return float(np.mean(scores)), float(np.std(scores))
    except Exception as e:
        print(f"    Cross-correlation failed: {e}")
        return None, None


def compute_discriminative(real, fake):
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
    print(f"  Computing metrics for {label}...")
    metrics = {}
    for name, fn in [("Context FID", compute_context_fid),
                     ("Cross Corr.", compute_cross_corr),
                     ("Discriminative", compute_discriminative),
                     ("Predictive", compute_predictive)]:
        m, s = fn(real, fake)
        metrics[name] = (m, s)
        if m is not None:
            print(f"    {name}: {m:.4f} +/- {s:.4f}")
    return metrics


# ── Table generation ─────────────────────────────────────────────────────────

def format_val(m, s, bold=False):
    """Format a metric value as mean +/- std."""
    if m is None:
        return "--"
    text = f"{m:.4f} +/- {s:.4f}"
    return text


def format_val_latex(m, s, bold=False):
    """Format a metric value for LaTeX with proper +/- symbol."""
    if m is None:
        return "--"
    core = rf"{m:.4f} \pm {s:.4f}"
    if bold:
        return rf"$\mathbf{{{core}}}$"
    return rf"${core}$"


def find_best(all_metrics, dataset, metric_name, methods):
    """Find the method with the lowest (best) mean for a given metric."""
    best_method = None
    best_val = float("inf")
    for method in methods:
        if (dataset in all_metrics
                and method in all_metrics[dataset]
                and metric_name in all_metrics[dataset][method]):
            m, _ = all_metrics[dataset][method][metric_name]
            if m is not None and m < best_val:
                best_val = m
                best_method = method
    return best_method


def generate_latex_table(all_metrics, methods, out_path):
    """Generate a LaTeX table: datasets as rows, metrics as columns, methods grouped."""
    datasets = [d for d in all_metrics if all_metrics[d]]
    if not datasets:
        print("  No metrics available for table generation.")
        return

    n_methods = len(methods)
    n_metrics = len(METRIC_NAMES)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Quantitative comparison of time series generation methods. "
                 r"All metrics are reported as mean $\pm$ std over 3 runs. "
                 r"\textbf{Bold} indicates the best result ($\downarrow$ lower is better).}")
    lines.append(r"  \label{tab:metrics}")

    # Column spec: Dataset | Method | metric1 | metric2 | ...
    col_spec = "ll" + "c" * n_metrics
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Header
    metric_headers = " & ".join(METRIC_LATEX[m] for m in METRIC_NAMES)
    lines.append(rf"    Dataset & Method & {metric_headers} \\")
    lines.append(r"    \midrule")

    for ds_idx, ds in enumerate(datasets):
        ds_display = DATASET_DISPLAY.get(ds, ds)

        for m_idx, method in enumerate(methods):
            method_label = "Diffusion-TS" if method == "diffts" else "PaD-TS"

            # Use multirow for dataset name on first method row
            if m_idx == 0 and n_methods > 1:
                ds_cell = rf"\multirow{{{n_methods}}}{{*}}{{{ds_display}}}"
            elif m_idx == 0:
                ds_cell = ds_display
            else:
                ds_cell = ""

            vals = []
            for metric_name in METRIC_NAMES:
                best = find_best(all_metrics, ds, metric_name, methods)
                is_bold = (best == method)

                if (method in all_metrics.get(ds, {})
                        and metric_name in all_metrics[ds][method]):
                    m, s = all_metrics[ds][method][metric_name]
                    vals.append(format_val_latex(m, s, bold=is_bold))
                else:
                    vals.append("--")

            val_str = " & ".join(vals)
            lines.append(rf"    {ds_cell} & {method_label} & {val_str} \\")

        if ds_idx < len(datasets) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"  Saved LaTeX table: {out_path}")

    return latex_str


def generate_csv_table(all_metrics, methods, out_path):
    """Generate a CSV table for easy import into slides/sheets."""
    datasets = [d for d in all_metrics if all_metrics[d]]
    if not datasets:
        return

    lines = []
    header = ["Dataset", "Method"] + METRIC_NAMES
    lines.append(",".join(header))

    for ds in datasets:
        ds_display = DATASET_DISPLAY.get(ds, ds)
        for method in methods:
            method_label = "Diffusion-TS" if method == "diffts" else "PaD-TS"
            row = [ds_display, method_label]
            for metric_name in METRIC_NAMES:
                if (method in all_metrics.get(ds, {})
                        and metric_name in all_metrics[ds][method]):
                    m, s = all_metrics[ds][method][metric_name]
                    row.append(format_val(m, s))
                else:
                    row.append("")
            lines.append(",".join(row))

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved CSV table: {out_path}")


def generate_per_dataset_latex(all_metrics, methods, out_dir):
    """Generate one small LaTeX table per dataset (useful for slides)."""
    datasets = [d for d in all_metrics if all_metrics[d]]

    for ds in datasets:
        ds_display = DATASET_DISPLAY.get(ds, ds)
        n_methods = len([m for m in methods if m in all_metrics.get(ds, {})])
        if n_methods == 0:
            continue

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(rf"  \caption{{{ds_display}: generation quality metrics "
                     r"($\downarrow$ lower is better).}}")
        lines.append(rf"  \label{{tab:{ds}_metrics}}")

        col_spec = "l" + "c" * n_methods
        lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \toprule")

        method_headers = []
        for method in methods:
            if method in all_metrics.get(ds, {}):
                method_headers.append("Diffusion-TS" if method == "diffts" else "PaD-TS")
        lines.append(r"    Metric & " + " & ".join(method_headers) + r" \\")
        lines.append(r"    \midrule")

        for metric_name in METRIC_NAMES:
            best = find_best(all_metrics, ds, metric_name, methods)
            row_label = METRIC_LATEX[metric_name]
            vals = []
            for method in methods:
                if method not in all_metrics.get(ds, {}):
                    continue
                if metric_name in all_metrics[ds][method]:
                    m, s = all_metrics[ds][method][metric_name]
                    is_bold = (best == method)
                    vals.append(format_val_latex(m, s, bold=is_bold))
                else:
                    vals.append("--")
            lines.append(rf"    {row_label} & " + " & ".join(vals) + r" \\")

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

        out_path = os.path.join(out_dir, f"{ds}_metrics.tex")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved: {out_path}")


def print_summary_table(all_metrics, methods):
    """Print a formatted table to the terminal."""
    datasets = [d for d in all_metrics if all_metrics[d]]
    if not datasets:
        return

    # Column widths
    ds_w = max(len(DATASET_DISPLAY.get(d, d)) for d in datasets)
    meth_w = max(len("Diffusion-TS"), len("PaD-TS"))
    val_w = 22  # "0.1234 +/- 0.1234"

    header = f"{'Dataset':<{ds_w}}  {'Method':<{meth_w}}"
    for mn in METRIC_NAMES:
        header += f"  {mn:>{val_w}}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for ds in datasets:
        ds_display = DATASET_DISPLAY.get(ds, ds)
        for m_idx, method in enumerate(methods):
            if method not in all_metrics.get(ds, {}):
                continue
            method_label = "Diffusion-TS" if method == "diffts" else "PaD-TS"
            ds_cell = ds_display if m_idx == 0 else ""
            row = f"{ds_cell:<{ds_w}}  {method_label:<{meth_w}}"
            for metric_name in METRIC_NAMES:
                if metric_name in all_metrics[ds][method]:
                    m, s = all_metrics[ds][method][metric_name]
                    val = format_val(m, s) if m is not None else "--"
                else:
                    val = "--"
                row += f"  {val:>{val_w}}"
            print(row)
        if ds != datasets[-1]:
            print()

    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate results tables")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all'")
    parser.add_argument("--output_dir", type=str, default="OUTPUT",
                        help="Directory containing .npy outputs")
    parser.add_argument("--table_dir", type=str, default="tables",
                        help="Directory to save tables (default: tables)")
    parser.add_argument("--padts_path", type=str, default=None,
                        help="Path to PaD-TS .npy file")
    parser.add_argument("--save_metrics", type=str, default=None,
                        help="Save metrics to JSON file")
    parser.add_argument("--load_metrics", type=str, default=None,
                        help="Load pre-computed metrics from JSON")
    args = parser.parse_args()

    os.makedirs(args.table_dir, exist_ok=True)

    if args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]

    # Either load or compute metrics
    if args.load_metrics and os.path.exists(args.load_metrics):
        print(f"Loading metrics from {args.load_metrics}")
        with open(args.load_metrics) as f:
            all_metrics = json.load(f)
        # Convert lists back to tuples
        for ds in all_metrics:
            for method in all_metrics[ds]:
                for metric in all_metrics[ds][method]:
                    v = all_metrics[ds][method][metric]
                    if isinstance(v, list):
                        all_metrics[ds][method][metric] = tuple(v)
    else:
        all_metrics = {}
        for ds in datasets:
            if ds not in DATASETS:
                print(f"Unknown dataset: {ds}")
                continue

            print(f"\n{'='*60}")
            print(f"  {DATASET_DISPLAY.get(ds, ds).upper()}")
            print(f"{'='*60}")

            data = load_dataset(ds, output_dir=args.output_dir, padts_path=args.padts_path)

            if "real" not in data or "diffts" not in data:
                print(f"  Skipping {ds}: missing data.")
                continue

            all_metrics[ds] = {}
            all_metrics[ds]["diffts"] = compute_all_metrics(
                data["real"], data["diffts"], "Diffusion-TS")
            if "padts" in data:
                all_metrics[ds]["padts"] = compute_all_metrics(
                    data["real"], data["padts"], "PaD-TS")

    # Save metrics if requested
    if args.save_metrics:
        save_dir = os.path.dirname(args.save_metrics)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        # Convert tuples to lists for JSON
        json_metrics = {}
        for ds in all_metrics:
            json_metrics[ds] = {}
            for method in all_metrics[ds]:
                json_metrics[ds][method] = {}
                for metric, val in all_metrics[ds][method].items():
                    json_metrics[ds][method][metric] = list(val) if isinstance(val, tuple) else val
        with open(args.save_metrics, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"\nSaved metrics to {args.save_metrics}")

    # Determine which methods are present
    methods = ["diffts"]
    if any("padts" in all_metrics.get(ds, {}) for ds in all_metrics):
        methods.append("padts")

    # Generate outputs
    print(f"\n{'='*60}")
    print("  Generating Tables")
    print(f"{'='*60}")

    # Terminal summary
    print_summary_table(all_metrics, methods)

    # Combined LaTeX table
    generate_latex_table(all_metrics, methods,
                         os.path.join(args.table_dir, "metrics_all.tex"))

    # Per-dataset LaTeX tables (for slides)
    generate_per_dataset_latex(all_metrics, methods, args.table_dir)

    # CSV
    generate_csv_table(all_metrics, methods,
                       os.path.join(args.table_dir, "metrics_all.csv"))

    print(f"\nDone! Tables saved to: {args.table_dir}/")


if __name__ == "__main__":
    main()
