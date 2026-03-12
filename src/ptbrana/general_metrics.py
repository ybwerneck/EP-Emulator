import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the custom color palette with distinct groups
blue_colors = sns.color_palette("Blues", 6)[3:]  # Darker blues
green_colors = sns.color_palette("Greens", 3)[:]  # Default greens
yellow_colors = sns.color_palette("Reds", 6)[3:]  # Default yellows
custom_palette = blue_colors + green_colors + yellow_colors

# List of problems
probs = ["A", "B"]

# Initialize global limits for consistent scaling across problems
global_limits = {
    "MARE": {"min": 5e-4, "max": 5e-1},
    "R2": {"min": 0.995, "max": 1.00001},
    "Inference Time (s)": {"min": 1e-3, "max": 10},
    "Training Time (s)": {"min": 1e-2, "max": 1e4},
    "Memory (gpu)": {"min": 1e-1, "max": 1e4},
}

# Metrics to process
metrics = list(global_limits.keys())

# Read and process data for all problems
data_dict = {prob: pd.read_csv(f'Results/inference_{prob}.csv') for prob in probs}

# Generate figures for each metric
for metric in metrics:
    fig, axes = plt.subplots(len(probs), 1, figsize=(12, 6 * len(probs)), sharex=True)

    for i, prob in enumerate(probs):
        ax = axes[i]
        data = data_dict[prob]
        data['Model'] = data['Model'].astype(str)  # Ensure 'Model' is string

        # Create a simple sequential list of model names for display (e.g., Model 1, Model 2, etc.)
        display_names = ["NN_S", "NN_M", "NN_G", "GP_S", "GP_M", "GP_G", "PC_2", "PC_3", "PC_5"]
        model_to_display = dict(zip(data['Model'].unique(), display_names))
        data['Display Model'] = data['Model'].map(model_to_display)

        # Assign colors based on model groups
        num_models = len(data['Display Model'].unique())
        custom_colors = custom_palette[:num_models]

        # Plot for the specific problem using the display names
        sns.barplot(data=data, x='Display Model', y=metric, palette=custom_colors, ax=ax, linewidth=2)

        # Increase font sizes and line widths
        ax.set_yscale('log' if metric != "R2" else 'linear')
        ax.set_ylim(global_limits[metric]["min"], global_limits[metric]["max"])
        ax.tick_params(axis='y', labelsize=20)

        # Set only the last subplot's x-axis label
        if i == len(probs) - 1:
            ax.set_xlabel('Model', fontsize=35)
        else:
            ax.set_xlabel('')

        ax.tick_params(axis='x', rotation=45, labelsize=35)

        # Set ylabel for the first subplot only
        
        ax.set_ylabel(metric, fontsize=30)

    # Add overall title for the figure
    plt.tight_layout()
    plt.subplots_adjust()  # To prevent overlap with the suptitle

    # Save the figure as a high-res PDF
    plt.savefig(f"Results/{metric}_comparison_across_problems.pdf", dpi=300)
    plt.close(fig)
