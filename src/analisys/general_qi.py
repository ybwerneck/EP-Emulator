import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_qoi_statistics(data):
    """
    Compute mean, standard deviation, and coefficient of variation per QoI.
    """
    stats = []
    print(data.columns)
    for col in data.columns:
        mean = data[col].mean()
        std = data[col].std()
        cv = std / abs(mean) if mean != 0 else np.nan
        stats.append({
            "QoI": col,
            "mean": mean,
            "std": std,
            "cv": cv
        })
    return pd.DataFrame(stats)


def infer_bounds(series, lower_q=0.01, upper_q=0.99):
    """
    Infer robust plotting bounds using quantiles.
    """
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def plot_qoi_histograms(data, output_path, num_bins=50):
    """
    Plot one histogram per QoI using data-driven bounds.
    """
    num_qois = data.shape[1]
    fig, axes = plt.subplots(
        1, num_qois,
        figsize=(5 * num_qois, 5),
        squeeze=False
    )

    for i, col in enumerate(data.columns):
        ax = axes[0, i]
        lo, hi = infer_bounds(data[col])
        bins = np.linspace(lo, hi, num_bins + 1)

        ax.hist(
            data[col],
            bins=bins,
            edgecolor="black"
        )

        ax.set_title(col, fontsize=14)
        ax.set_xlim(lo, hi)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def analyze_folder(folder_path):
    """
    Perform QoI analysis for a single folder containing X.csv and Y.csv.
    """
    x_path = os.path.join(folder_path, "X.csv")
    y_path = os.path.join(folder_path, "Y.csv")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return

    print(f"Analyzing folder: {folder_path}")

    Y = pd.read_csv(y_path)

   # Y = Y.iloc[:, :-1]

    # Statistics table
    stats_df = compute_qoi_statistics(Y)
    stats_path = os.path.join(folder_path, "QoI_statistics.csv")
    stats_df.to_csv(stats_path, index=False)

    # Histogram plot
    hist_path = os.path.join(folder_path, "QoI_histograms.png")
    plot_qoi_histograms(Y, hist_path)

    print(f"Saved statistics to {stats_path}")
    print(f"Saved histograms to {hist_path}")


def process_root_directory(root_dir):
    """
    Walk directory tree and analyze every folder containing X.csv and Y.csv.
    """
    for dirpath, _, _ in os.walk(root_dir):
        try:
            analyze_folder(dirpath)
        except Exception as e:
            print(f"Error analyzing folder {dirpath}: {e}")


# Root directory
root_directory = "./"

# Run analysis
process_root_directory(root_directory)
