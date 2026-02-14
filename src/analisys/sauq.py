import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Utilities
# -----------------------------
def extract_training_samples(name):
    part = name.split('_')[-1]          # e.g. "0.1K.pth"
    k_val = part.replace('K','').replace('.pth','').replace('.pkl','')
    return int(float(k_val) * 1000)

def get_size(n):
    if n == 100:  return 60
    if n == 200:  return 120
    if n == 500:  return 200
    if n == 1000: return 350
    return 100

# -----------------------------
# Color palette
# -----------------------------
blue_colors  = sns.color_palette("Blues", 6)[3:]
green_colors = sns.color_palette("Greens", 3)
reds_colors  = sns.color_palette("Reds", 6)[3:]
custom_palette = blue_colors + green_colors + reds_colors

model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}

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

    def get_subtype(name):
        if '_mc' in name: return 'mc'
        return 'default'

    uqsa_df['Model'] = uqsa_df['model'].apply(map_model_name)
    uqsa_df['subtype'] = uqsa_df['model'].apply(get_subtype)
    uqsa_df['Training Samples'] = uqsa_df['model'].apply(extract_training_samples)

    uqsa_dict[prob] = uqsa_df[['Model','Training Samples','subtype','mean_rel','S1_rel','model']]
    uqsa_dict[prob] = uqsa_dict[prob][~uqsa_dict[prob]['model'].str.contains('basis', na=False)]

# -----------------------------
# Plot metrics with linear family regressions
# -----------------------------
for metric_name in ['mean_rel','S1_rel']:

    n_probs = len(probs)
    cols = 2
    rows = int(np.ceil(n_probs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6),sharey=False, sharex=True)
    axes = axes.flatten() if n_probs > 1 else [axes]

    for idx, prob in enumerate(probs):
        ax = axes[idx]

        data = data_dict[prob].copy()
        data['Model'] = data['Model'].astype(str)

        # Pair UQ/SA with (Model, Training Samples)
        merged_df = data.merge(
            uqsa_dict[prob],
            on=['Model', 'Training Samples'],
            how='left'
        )

        merged_df['size'] = merged_df['Training Samples'].apply(get_size)

        # -----------------------------
        # Scatter
        # -----------------------------
        for _, row in merged_df.iterrows():
            ax.scatter(
                x=row['MARE'],
                y=row[metric_name],
                s=row['size'],
                color=color_mapping[row['Model']],
                edgecolor='black',
                alpha=0.8,
                zorder=3
            )

        # -----------------------------
        # Broad-family linear regressions
        # -----------------------------
        family_map = {
            'NN':  ['NN_S','NN_M','NN_L'],
            'GP':  ['gp_S','gp_M','gp_L'],
            'PCE': ['PCE_2','PCE_3','PCE_5']
        }

        family_colors = {
            'NN':  sns.color_palette("Blues", 6)[4],
            'GP':  sns.color_palette("Greens", 6)[4],
            'PCE': sns.color_palette("Reds", 6)[4]
        }

        for family, models in family_map.items():

            sub = merged_df[merged_df['Model'].isin(models)].dropna(subset=['MARE', metric_name])

            if len(sub) < 2:
                continue

            x = sub['MARE'].values
            y = sub[metric_name].values

            # linear regression y = a x + b
            a, b = np.polyfit(x, y, 1)

            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = a * x_line + b

            ax.plot(
                x_line, y_line,
                color=family_colors[family],
                linewidth=3,
                alpha=0.75,
                zorder=0
            )

            # annotation
            xm = np.mean(x)
            ym = a * xm + b

            ax.text(
                xm, ym,
                f"{family}:{a:.1f}x + {b:.3f}",
                fontsize=18,
                color=family_colors[family],
                ha='left',
                va='bottom'
            )
            ax.set_xscale('log')
            ax.set_yscale('log')

        # -----------------------------
        # Axes formatting
        # -----------------------------
        ax.set_xlabel('MARE', fontsize=18)
        ylabel = 'UQ Error' if metric_name=='mean_rel' else 'S1 Relative Error'
        ax.set_xlabel('MARE', fontsize=18)

        if idx % cols == 0:   # only left subplot gets y-label
            ylabel = 'UQ Error' if metric_name=='mean_rel' else 'S1 Relative Error'
            ax.set_ylabel(ylabel, fontsize=18)
        else:
            ax.set_ylabel('')



        ax.set_title(f'Problem {prob}', fontsize=18)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    # Remove empty axes
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    # -----------------------------
    # Legend
    # -----------------------------
    color_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]
    plt.subplots_adjust(wspace=0.03, hspace=0.01)

    fig.legend(
        handles=color_handles,
        title='Model',
        fontsize=18,
        title_fontsize=18,
        loc='center left',
        bbox_to_anchor=(0.99, 0.5)
    )

    plt.tight_layout()
    plt.savefig(f'Results/plots/{metric_name}_family_linear_scatter.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
