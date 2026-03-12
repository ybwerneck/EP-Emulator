

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_qoi_histograms_for_two_models(file_paths):
    # Define the scale for each QoI
    scales = [
        (0, 2),      # V_peak: allow extreme depolarization/hyperpolarization
        (0, 10),       # V_rest: very negative resting potentials
        (0, 1),    # Ca_peak: large spread around SR/Ca transients

        (0, 100),
        (100, 130),        # APD90/50: much longer durations (e.g., diseased models)
        (0, 1),        # APD30: same
    ]
    # Number of QoIs to plot (excluding the last one)
    num_qois = len(scales)

    # Create a figure with two rows (one for each model) and columns for each QoI
    num_cols = num_qois  # Columns are based on the number of QoIs
    num_rows =2  # Two rows, one for each model

    # Increase the size of the plot for clarity, and set sharex=True to share the x-axis for the same QoIs
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 16))  # Adjust figure size accordingly
    axes = axes.flatten()

    # Loop through the file paths for the two models
    for model_idx, file_path in enumerate(file_paths):
        data = pd.read_csv(file_path)

        # Exclude the last QoI (last column) from the analysis
        data = data.iloc[:, :-1]

        # Define the number of bins for each QoI
        num_bins = 50  # Fixed number of bins
        n = ["A","B"]
        
        for i, column in enumerate(data.columns):
            # Generate bin edges based on the scale and the number of bins
            bin_edges = np.linspace(scales[i][0], scales[i][1], num_bins + 1)
            
            # Plot the histogram for the QoI of this model
            color = 'b' if model_idx == 0 else 'r' if model_idx == 1 else 'g'
              # Standard blue and red from matplotlib
            axes[model_idx * num_qois + i].hist(data[column], bins=bin_edges, alpha=0.75, color=color, edgecolor='black', label=f'{column}\nModel {n[model_idx]}')
            
            # Improve the title and labels with larger font sizes
          #  if(n[model_idx] == "B"):
           #     axes[model_idx * num_qois + i].set_xlabel(f'QoI {column}', fontsize=24)
            
            # Remove y-axis tick labels for all but the first column
            axes[model_idx * num_qois + i].tick_params(axis='y', labelleft=False)

            # Increase the font size for tick labels
            axes[model_idx * num_qois + i].tick_params(axis='both', labelsize=24)

            # Set grid and adjust limits
            axes[model_idx * num_qois + i].set_xlim(scales[i])
            axes[model_idx * num_qois + i].set_ylim(0, int(data.shape[0] * 0.15))  # Adjust y limit

            # Set ticks near the center of the range (midpoint of the first, middle, and last bin edges)
            center_tick_positions = [
                (bin_edges[3] + bin_edges[4]) / 2,  # Middle of the first bin
                (bin_edges[len(bin_edges) // 2 ] + bin_edges[len(bin_edges) // 2 +2]) / 2,  # Middle of the central bin
                (bin_edges[-2] + bin_edges[-1]) / 2  # Middle of the last bin
            ]
            axes[model_idx * num_qois + i].set_xticks(center_tick_positions)

            # Increase the font size of the legend
            axes[model_idx * num_qois + i].legend(loc='upper right', fontsize=24)

            # Rotate the x-tick labels to avoid overlap
            plt.setp(axes[model_idx * num_qois + i].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout for better spacing, reducing space between subplots
    plt.subplots_adjust(hspace=0., wspace=0.)  # Reduce vertical and horizontal space

    # Save the combined histogram plot
    combined_histogram_path = file_paths[0].replace("Y.csv", "Combined_Model_QoI_histograms.png")
    plt.savefig(combined_histogram_path, bbox_inches='tight')  # Save with tight bounding box to remove excess margin
    plt.close()
    print(f"Saved combined histogram for all models to {combined_histogram_path}")

def process_directory_for_histograms(root_dir):
    # Walk through all directories and subdirectories and collect Y.csv paths for Model A and Model B subfolders
    for model_dir in os.walk(root_dir):
        model_path = model_dir[0]
        model_a_path = os.path.join(model_path, "Ho8", "Y.csv")
        model_b_path = os.path.join(model_path, "Tiso", "Y.csv")
        print(model_path)
        # Check if both Model A and Model B have a Y.csv file
        if os.path.exists(model_a_path) and os.path.exists(model_b_path) :
            file_paths = [model_a_path, model_b_path]
            plot_qoi_histograms_for_two_models(file_paths)

# Define the root directory containing "Generated_Data_XK"
root_directory = './'

# Generate histograms for all Y.csv files in the directory tree
process_directory_for_histograms(root_directory)
