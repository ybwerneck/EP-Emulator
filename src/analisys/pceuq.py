import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np
# -----------------------------
# Color palette for PCE models
# -----------------------------
pce_models = ["PCE_2", "PCE_3", "PCE_5"]
colors = sns.color_palette("Reds", len(pce_models))
color_mapping = {m: c for m, c in zip(pce_models, colors)}
marker_mapping = {'basis':'o', 'mc':'X'}

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
        return name

    uqsa_df['Model'] = uqsa_df['model'].apply(map_model_name)

    def get_subtype(name):
        if '_basis' in name: return 'basis'
        if '_mc' in name: return 'mc'
        return 'default'

    uqsa_df['subtype'] = uqsa_df['model'].apply(get_subtype)
    uqsa_dict[prob] = uqsa_df[['Model','subtype','mean_rel','S1_rel','uq_time_s','sa_time_s','model']]

# -----------------------------
# Plot PCE accuracy vs cost
# -----------------------------
for metric_name, ylabel in [('mean_rel','Mean Relative Error (Accuracy)'), ('S1_rel','S1 Relative Error')]:
    fig, axes = plt.subplots(1, len(probs), figsize=(12,6), sharey=True,sharex=True)
    axes = axes.flatten() if len(probs) > 1 else [axes]

    for idx, prob in enumerate(probs):
        ax = axes[idx]
        df = uqsa_dict[prob]
        df = df[df['Model'].isin(pce_models)]  # Only PCE

        for _, row in df.iterrows():
            # Cost: combine UQ + SA time
            cost = row['uq_time_s'] + row['sa_time_s']

            ax.scatter(
                x=cost,
                y=row[metric_name],
                color=color_mapping[row['Model']],
                marker=marker_mapping.get(row['subtype'],'o'),
                s=150,
                edgecolor='black',
                alpha=0.8
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Total Time [s]', fontsize=18)
        ylabel = 'UQ Error' if metric_name=='mean_rel' else 'S1 Relative Error'
        from matplotlib.ticker import LogLocator

        # inside the subplot loop, after setting log scale
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=2))  # only major powers of 10
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=4))  # optional minor ticks

        ax.set_ylabel(ylabel if idx==0 else '', fontsize=18)
        ax.set_title(f'Problem {prob}', fontsize=18)
     #   ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.25)

    # Legend
    color_handles = [Patch(color=color_mapping[m], label=m) for m in pce_models]
    marker_handles = [plt.Line2D([0],[0], marker='o', color='k', linestyle='', label='basis', markersize=10),
                      plt.Line2D([0],[0], marker='X', color='k', linestyle='', label='MC', markersize=10)]

    fig.legend(handles=color_handles + marker_handles,
               title='Model / Type',
               fontsize=12,
               title_fontsize=12,
               loc='center right',
               bbox_to_anchor=(0.97,0.5))

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'Results/plots/pce_accuracy_vs_cost_{metric_name}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
