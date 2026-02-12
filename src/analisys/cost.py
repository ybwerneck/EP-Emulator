
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
probs = ["A", "B"]

# -----------------------------
    # Load data
    # -----------------------------
    data_dict = {prob: pd.read_csv(f'Results/inference_{prob}.csv') for prob in probs}

# -----------------------------
# Layout
# -----------------------------
n_probs = len(probs)
cols = 2
rows = int(np.ceil(n_probs / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
axes = axes.flatten() if n_probs > 1 else [axes]

# -----------------------------
# Plot
# -----------------------------
for idx, prob in enumerate(probs):
    ax = axes[idx]
    data = data_dict[prob].copy()
    data['Model'] = data['Model'].astype(str)

    # Required columns:
    # 'Training Time (s)'
    # 'Inference Time (s)'

    present_models = sorted(
        data['Model'].unique(),
        key=lambda x: model_names.index(x)
    )
    present_colors = {m: color_mapping[m] for m in present_models}

    sns.scatterplot(
        data=data,
        x='Training Time (s)',
        y='Inference Time (s)',
        hue='Model',
        palette=present_colors,
        s=200,
        alpha=0.6,
        edgecolor='none',
        ax=ax,
        legend=False
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(f'Problem {prob}', fontsize=16)
    ax.set_xlabel('Training time (s)', fontsize=14)
    ax.set_ylabel('Inference time per sample (s)', fontsize=14)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.25)

# -----------------------------
# Remove empty axes
# -----------------------------
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

# -----------------------------
# Legend
# -----------------------------
legend_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]

fig.legend(
    handles=legend_handles,
    title='Emulador',
    fontsize=12,
    title_fontsize=12,
    loc='center right',
    bbox_to_anchor=(0.995, 0.5)
)

# -----------------------------
# Save
# -----------------------------
plt.tight_layout()
plt.subplots_adjust(right=0.88)
plt.savefig('Results/plots/compute_2d_distribution.png', dpi=600, bbox_inches='tight')
# plt.show()