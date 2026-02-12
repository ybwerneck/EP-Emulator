import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Color palette
# -----------------------------
blue_colors  = sns.color_palette("Blues", 6)[3:]
green_colors = sns.color_palette("Greens", 3)
reds_colors  = sns.color_palette("Reds", 6)[3:]
custom_palette = blue_colors + green_colors + reds_colors

model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}
marker_mapping = {'basis':'o', 'mc':'X', 'default':'o'}

probs = ["A","B"]

# -----------------------------
# Load data
# -----------------------------
data_dict = {prob: pd.read_csv(f'Results/inference_{prob}.csv') for prob in probs}
uqsa_dict = {}

for prob in probs:
    uqsa_df = pd.read_csv(f'Results/uq_sa_{prob}.csv')

    def map_model_name(name):
        if "pce_model2" in name: return "PCE_2"
        if "pce_model3" in name: return "PCE_3"
        if "pce_model5" in name: return "PCE_5"
        if "nmodel_S" in name: return "NN_S"
        if "nmodel_M" in name: return "NN_M"
        if "nmodel_L" in name: return "NN_L"
        if "gp_S" in name: return "gp_S"
        if "gp_M" in name: return "gp_M"
        if "gp_L" in name: return "gp_L"
        return name

    uqsa_df['Model'] = uqsa_df['model'].apply(map_model_name)

    def get_subtype(name):
       # if '_basis' in name: return 'basis'
        if '_mc' in name: return 'mc'
        return 'default'

    uqsa_df['subtype'] = uqsa_df['model'].apply(get_subtype)
    uqsa_dict[prob] = uqsa_df[['Model','subtype','mean_rel','S1_rel','model']]  # keep model for size mapping


    uqsa_dict[prob] = uqsa_dict[prob][~uqsa_dict[prob]['model'].str.contains('basis', na=False)]
# -----------------------------
# Plot metrics with scatter sizes
# -----------------------------
for metric_name in ['mean_rel','S1_rel']:
    n_probs = len(probs)
    cols = 2
    rows = int(np.ceil(n_probs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
    axes = axes.flatten() if n_probs > 1 else [axes]

    for idx, prob in enumerate(probs):
        ax = axes[idx]
        data = data_dict[prob].copy()
        data['Model'] = data['Model'].astype(str)

        merged_df = data.merge(uqsa_dict[prob], on='Model', how='left')

        # Map training set size to marker size
        # Example: parse training size from the filename if present, else use 1K -> 100, 10K -> 200, etc.
        def get_size(name):
            if 100 == name: return 60
            if 200 == name: return 120
            if 500 ==name: return 200
            if 1000 == name: return 350
            return 100



        merged_df['Training Samples'] = merged_df['Training Samples'].apply(get_size)

        # Scatter points
        for _, row in merged_df.iterrows():
            ax.scatter(
                x=row['MARE'],
                y=row[metric_name],
                s=row['Training Samples'],
                color=color_mapping[row['Model']],
               # marker=marker_mapping.get(row['subtype'],'o'),
                edgecolor='black',
                alpha=0.8
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('MARE', fontsize=14)
        ylabel = 'UQ Error' if metric_name=='mean_rel' else 'S1 Relative Error'
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'Problem {prob}', fontsize=16)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.25)

    # Remove empty axes
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    # Legend
    color_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]
   # marker_handles = [plt.Line2D([0],[0], marker='o', color='k', linestyle='', label='basis', markersize=10),
      #                plt.Line2D([0],[0], marker='X', color='k', linestyle='', label='MC', markersize=10)]

    fig.legend(handles=color_handles,
               title='Emulador / PCE subtype',
               fontsize=12,
               title_fontsize=12,
               loc='center right',
               bbox_to_anchor=(0.95, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    plt.savefig(f'Results/plots/{metric_name}_all_problems_scatter.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
