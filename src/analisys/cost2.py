import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Color palette
# -----------------------------
blue_colors  = sns.color_palette("Blues", 6)[3:]
green_colors = sns.color_palette("Greens", 3)[:]
reds_colors  = sns.color_palette("Reds", 6)[3:]
custom_palette = blue_colors + green_colors + reds_colors 

model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}

# -----------------------------
# Problems
# -----------------------------
probs = ["A", "B", "C", "D"]
# Map problem codes to LaTeX names for titles and legend
prob_labels = {
    "A": r"$A_E$",
    "B": r"$B_E$",
    "C": r"$A_M$",
    "D": r"$B_M$"
}

factor=1e5/500
# -----------------------------
# True model inference times (s) per problem
true_inference = {
    "A": 40.05,
    "B": 40.1061,
    "C": 56191,
    "D": 68453,
}
true_inference = {k: v*factor for k, v in true_inference.items()}   

# -----------------------------
# Load data
# -----------------------------
data_dict = {prob: pd.read_csv(f'Results/inference_{prob}.csv') for prob in probs}

# -----------------------------
# Marker size mapping (training set size)
# -----------------------------
size_map = {
    100: 60,
    200: 120,
    500: 200,
    1000: 350
}

# -----------------------------
# Layout
# -----------------------------
n_probs = len(probs)
cols = 2
rows = int(np.ceil(n_probs / cols))
fig, axes = plt.subplots(
    rows, cols,
    figsize=(16, 6 * rows),
    sharex='col',   # x-axis shared within each column
    sharey='row'    # y-axis shared within each row
)
axes = axes.flatten() if n_probs > 1 else [axes]

# -----------------------------
# Plot each problem
# -----------------------------
for idx, prob in enumerate(probs):
    ax = axes[idx]
    data = data_dict[prob].copy()
    data['Model'] = data['Model'].astype(str)

    # Map marker sizes by training set size
    if 'Training Samples' in data.columns:
        data['marker_size'] = data['Training Samples'].map(size_map)*2
    else:
        data['marker_size'] = 150  # fallback

    present_models = sorted(
        data['Model'].unique(),
        key=lambda x: model_names.index(x)
    )
    present_colors = {m: color_mapping[m] for m in present_models}
    data['Inference Time (s)'] = data['Inference Time (s)'] * 100
    sns.scatterplot(
        data=data,
        x='Training Time (s)',
        y='Inference Time (s)',
        hue='Model',
        palette=present_colors,
        size='marker_size',
        sizes=(60, 350),
        alpha=0.80,
        edgecolor='black',
        linewidth=0.3,
        ax=ax,
        legend=False
    )

    # Add horizontal dashed line for true model inference
    ax.axhline(
        y=true_inference[prob],
        color='black',
        linestyle='--',
        linewidth=2,
        label='True model'
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(prob_labels[prob], fontsize=18)
    ax.set_xlabel('Training time (s)', fontsize=18)
    ax.set_ylabel('Inference time for 100K sample (s)', fontsize=18)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.25)

# Remove empty axes
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Training and Inference Time Scaling", fontsize=20)

# -----------------------------
# Legends
# -----------------------------

# Model legend
model_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]
fig.legend(
    handles=model_handles,
    title='Emulator',
    fontsize=18,
    title_fontsize=18,
    loc='center right',
    bbox_to_anchor=(0.999, 0.55)
)

# Training-size legend (marker size)
size_handles = [
    plt.scatter([], [], s=size_map[s], edgecolors='black', facecolors='gray', alpha=0.6, label=f'{s}')
    for s in size_map
]
fig.legend(
    handles=size_handles,
    title='Set size',
    fontsize=18,
    title_fontsize=18,
    loc='center right',
    bbox_to_anchor=(0.999, 0.25)
)

# True model legend
true_line = plt.Line2D([], [], color='black', linestyle='--', linewidth=2, label='True model')
fig.legend(
    handles=[true_line],
    title='Reference',
    fontsize=18,
    title_fontsize=18,
    loc='center right',
    bbox_to_anchor=(0.999, 0.80)
)

# -----------------------------
# Save figure
# -----------------------------
plt.tight_layout()
plt.subplots_adjust(right=0.88)
plt.savefig('Results/plots/compute_training_inference_scaling.png', dpi=600, bbox_inches='tight')
# plt.show()
