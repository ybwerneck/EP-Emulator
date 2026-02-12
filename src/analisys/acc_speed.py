import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# Define the custom color palette with distinct groups
blue_colors = sns.color_palette("Blues", 6)[3:]  # Darker blues
green_colors = sns.color_palette("Greens", 3)[:]  # Default greens
reds_colors = sns.color_palette("Reds", 6)[3:]  # Darker reds
custom_palette = blue_colors + green_colors + reds_colors 

# List of problems
probs = ["A", "B"]  # Example: multiple problems

# Display names mapping
model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
# Map models to colors
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}

# Read and process data for all problems
data_dict = {prob: pd.read_csv(f'Results/inference_{prob}.csv') for prob in probs}

# Determine figure layout
n_probs = len(probs)
cols = 2
rows = int(np.ceil(n_probs / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
axes = axes.flatten() if n_probs > 1 else [axes]

for idx, prob in enumerate(probs):
    ax = axes[idx]
    data = data_dict[prob]
    data['Model'] = data['Model'].astype(str)

    # Compute overall MARE across all QoIs
    qoi_cols = [col for col in data.columns if col.startswith('MARE_QoI_')]
    data['MARE_overall'] = data[qoi_cols].mean(axis=1)

    # Map Display Model for coloring
    # Only include models present in this dataset
    present_models = sorted(data['Model'].unique(), key=lambda x: model_names.index(x))
    present_colors = {m: color_mapping[m] for m in present_models}
    
    # Scatter plot: force palette mapping
    sns.scatterplot(
        data=data, x='Inference Time (s)', y='MARE_overall',
        hue='Model', palette=present_colors, s=100,
        edgecolor='black', ax=ax, legend=False
    )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(f'Problem {prob}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Inference time (s)', fontsize=14)
    ax.set_ylabel('Overall MARE', fontsize=14)

# Remove empty axes if number of problems < grid size
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

# Create manual legend in the correct order
legend_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]

fig.legend(handles=legend_handles, title='Emulador', fontsize=12, title_fontsize=12,
           loc='center right', bbox_to_anchor=(0.995, 0.5))

plt.tight_layout()
plt.subplots_adjust(right=0.88)
plt.savefig('Results/plots/mare_inference_overall.png', dpi=600, bbox_inches='tight')
# plt.show()
