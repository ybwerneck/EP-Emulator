import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_qoi_subplots(folder_paths, output_path="combined_qoi_histograms.png"):
    """
    Generate a combined figure with one row per folder, showing histograms of all QoIs.
    
    Parameters:
        folder_paths : list of str
            List of folders, each containing a single Y.csv file.
        output_path : str
            Path to save the combined figure.
    """

    # Define expected QoI display ranges (adjust as needed)
    scales = [
        (-100, -70),      # V_peak: allow extreme depolarization/hyperpolarization
        (15, 40),       # V_rest: very negative resting potentials
        (49, 56),    # Ca_peak: large spread around SR/Ca transients

        (30, 350),
        (30, 350),        # APD90/50: much longer durations (e.g., diseased models)
        (30, 350),        # APD30: same
    ]

    n_folders = len(folder_paths)
    if n_folders == 0:
        print("No folders provided.")
        return

    # Read first file to determine number of QoIs
    first_file = os.path.join(folder_paths[0], "Y.csv")
    ref_data = pd.read_csv(first_file)
    n_qois = ref_data.shape[1]
    scales = scales[:n_qois]

    # Figure setup: one row per folder, one column per QoI
    fig_width = max(5 * n_qois, 10)
    fig_height = max(4 * n_folders, 6)
    fig, axes = plt.subplots(
        n_folders, n_qois, figsize=(fig_width, fig_height),
        sharey="row", sharex="col"
    )

    if n_folders == 1:
        axes = [axes]  # make iterable
    if n_qois == 1:
        axes = [[ax] for ax in axes]  # handle single QoI

    colors = plt.cm.tab10.colors
    num_bins = 12

    # Loop over folders
    for row_idx, folder in enumerate(folder_paths):
        file_path = os.path.join(folder, "Y.csv")
        if not os.path.exists(file_path):
            print(f"Y.csv not found in folder {folder}, skipping.")
            continue

        data = pd.read_csv(file_path)
        data = data.iloc[:, :n_qois]  # ensure correct number of QoIs

        for col_idx, column in enumerate(data.columns):
            ax = axes[row_idx][col_idx]
            bin_edges = np.linspace(scales[col_idx][0], scales[col_idx][1], num_bins + 1)
            ax.hist(
                data[column],
                bins=bin_edges,
                alpha=0.7,
                color=colors[row_idx % len(colors)],
                edgecolor="black",
                label=os.path.basename(folder)
            )
            ax.set_xlim(scales[col_idx])
            ax.set_yscale("log")
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(alpha=0.25)

            if col_idx == 0:
                ax.set_ylabel("Count (log scale)", fontsize=14)
            if row_idx == 0:
                ax.set_title(column, fontsize=16)
            if row_idx == n_folders - 1:
                ax.set_xlabel(column, fontsize=14)

            # Only show legend for first column to avoid clutter
            if col_idx == 0:
                ax.legend(fontsize=12)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved combined figure to {output_path}")


# ===========================================
# Example usage
# ===========================================
if __name__ == "__main__":
    folders = [
        "./Generated_Data_0.2K/ModelA",
        "./Generated_Data_0.2K/ModelB",
    ]
    plot_combined_qoi_subplots(folders, output_path="Results/combined_qoi_histograms.png")
