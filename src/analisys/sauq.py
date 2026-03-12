import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Utilities
# -----------------------------
def extract_training_samples(name):
    try:
        part = name.split('_')[-1]
        k_val = part.replace('K','').replace('.pth','').replace('.pkl','')
        return int(float(k_val) * 1000)
    except:
        part = name.split('_')[-2]
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
# Custom problem names
# -----------------------------
prob_names = {
    "A": "$A_E$",
    "B": "$B_E$"
}

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
        if '_mc' in name:
            return 'mc'
        return 'default'

    uqsa_df['Model'] = uqsa_df['model'].apply(map_model_name)
    uqsa_df['subtype'] = uqsa_df['model'].apply(get_subtype)
    uqsa_df['Training Samples'] = uqsa_df['model'].apply(extract_training_samples)

    uqsa_dict[prob] = uqsa_df[['Model','Training Samples','subtype',
                               'mean_rel','std_rel','S1_rel','ST_rel','model']]

    uqsa_dict[prob] = uqsa_dict[prob][~uqsa_dict[prob]['model'].str.contains('basis', na=False)]

# -----------------------------
# Metric groups
# -----------------------------
metric_groups = {
    'UQ': ['mean_rel','std_rel'],
    'SA': ['S1_rel','ST_rel']
}
# -----------------------------
# Linear model families
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

family_order = {'NN':0, 'GP':1, 'PCE':2}


# -----------------------------
# Plot
# -----------------------------
for group_name, metrics in metric_groups.items():

    rows = len(metrics)
    cols = len(probs)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(8,12),
        sharex=True,
        sharey='row'
    )

    if rows == 1:
        axes = np.array([axes])

    # Labels for rows (do NOT modify metrics)
    if group_name == 'UQ':
        metric_labels = ["MARE of UQ-Mean", "MARE of UQ-Std"]
    else:
        metric_labels = ["S1 Error", "ST Error"]

    # ----------------------------------
    # Loop over problems (columns)
    # ----------------------------------
    for j, prob in enumerate(probs):

        data = data_dict[prob].copy()
        data['Model'] = data['Model'].astype(str)

        merged_df = data.merge(
            uqsa_dict[prob],
            on=['Model', 'Training Samples'],
            how='left'
        )

        merged_df['size'] = merged_df['Training Samples'].apply(get_size)

        # ----------------------------------
        # Loop over metrics (rows)
        # ----------------------------------
        for i, metric_name in enumerate(metrics):

            ax = axes[i, j]

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
            # Linear regression families
            # -----------------------------
            for family, models in family_map.items():

                sub = merged_df[
                    merged_df['Model'].isin(models)
                ].dropna(subset=['MARE', metric_name])

                if len(sub) < 2:
                    continue

                x = sub['MARE'].values
                y = sub[metric_name].values

                a, b = np.polyfit(x, y, 1)

                x_line = np.linspace(x.min(), x.max(), 200)
                y_line = a * x_line + b

                ax.plot(
                    x_line,
                    y_line,
                    color=family_colors[family],
                    linewidth=3,
                    alpha=0.75
                )

                ypos = 0.01 + 0.06 * family_order[family]

                ax.text(
                    0.25,
                    ypos,
                    f"{family}: {a:.1f}x + {b:.2f}",
                    transform=ax.transAxes,
                    fontsize=14,
                    color=family_colors[family],
                    ha='left',
                    va='bottom'
                )

            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                alpha=0.25
            )

    # -----------------------------
    # Column titles = Problems
    # -----------------------------
    for j, prob in enumerate(probs):

        axes[0, j].set_title(
            f"Problem {prob_names[prob]}",
            fontsize=18
        )

    # -----------------------------
    # Row labels = Metrics
    # -----------------------------
    for i in range(rows):

        axes[i, 0].set_ylabel(
            metric_labels[i],
            fontsize=16
        )

    # -----------------------------
    # X labels
    # -----------------------------
    for j in range(cols):

        axes[-1, j].set_xlabel(
            "Validation MARE",
            fontsize=16
        )

    # -----------------------------
    # Legend
    # -----------------------------
    color_handles = [
        Patch(color=color_mapping[m], label=m)
        for m in model_names
    ]

    fig.legend(
        handles=color_handles,
        title='Model',
        fontsize=16,
        title_fontsize=16,
        loc='center left',
        bbox_to_anchor=(0.99, 0.5)
    )

    plt.subplots_adjust(
        wspace=0.05,
        hspace=0.08
    )

    plt.tight_layout()

    plt.savefig(
        f'Results/plots/{group_name}_family_linear_scatter.png',
        dpi=600,
        bbox_inches='tight'
    )

    plt.close(fig)