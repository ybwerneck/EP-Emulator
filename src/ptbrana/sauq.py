import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Funções auxiliares
# -----------------------------
def extrair_amostras_treinamento(name):
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
# Paleta de cores
# -----------------------------
blue_colors  = sns.color_palette("Blues", 6)[3:]
green_colors = sns.color_palette("Greens", 3)
reds_colors  = sns.color_palette("Reds", 6)[3:]
custom_palette = blue_colors + green_colors + reds_colors

model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}

probs = ["A","B"]

# -----------------------------
# Nomes customizados dos problemas
# -----------------------------
prob_names = {
    "A": "$A_E$",
    "B": "$B_E$"
}

# -----------------------------
# Carregamento dos dados
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
    uqsa_df['Training Samples'] = uqsa_df['model'].apply(extrair_amostras_treinamento)

    uqsa_dict[prob] = uqsa_df[['Model','Training Samples','subtype',
                               'mean_rel','std_rel','S1_rel','ST_rel','model']]

    uqsa_dict[prob] = uqsa_dict[prob][~uqsa_dict[prob]['model'].str.contains('basis', na=False)]

# -----------------------------
# Grupos de métricas
# -----------------------------
metric_groups = {
    'UQ': ['mean_rel','std_rel'],
    'SA': ['S1_rel','ST_rel']
}

# -----------------------------
# Layout das métricas
# -----------------------------
metric_layout = [
    ("MARE da Média (UQ)", "mean_rel"),
  #  ("MARE do Desvio Padrão (UQ)",  "std_rel"),
    ("MAE S1",        "S1_rel"),
#    ("Erro ST",        "ST_rel")
]

rows = len(metric_layout)
cols = len(probs)

fig, axes = plt.subplots(rows, cols, figsize=(12,6), sharex=True, sharey='row')

for r, (metric_label, metric_name) in enumerate(metric_layout):

    for c, prob in enumerate(probs):

        ax = axes[r, c]

        data = data_dict[prob].copy()
        data['Model'] = data['Model'].astype(str)

        merged_df = data.merge(
            uqsa_dict[prob],
            on=['Model','Training Samples'],
            how='left'
        )

        merged_df['size'] = merged_df['Training Samples'].apply(get_size)

        # -----------------------------
        # Scatter
        # -----------------------------
        for _, row in merged_df.iterrows():

            ax.scatter(
                row['MARE'],
                row[metric_name],
                s=row['size'],
                color=color_mapping[row['Model']],
                edgecolor='black',
                alpha=0.8,
                zorder=3
            )

        # -----------------------------
        # Regressões por família de modelo
        # -----------------------------
        family_map = {
            'NN':  ['NN_S','NN_M','NN_L'],
            'GP':  ['gp_S','gp_M','gp_L'],
            'PCE': ['PCE_2','PCE_3','PCE_5']
        }

        family_colors = {
            'NN':  sns.color_palette("Blues",6)[4],
            'GP':  sns.color_palette("Greens",6)[4],
            'PCE': sns.color_palette("Reds",6)[4]
        }

        family_order = {'NN':0,'GP':1,'PCE':2}

        for family, models in family_map.items():

            sub = merged_df[
                merged_df['Model'].isin(models)
            ].dropna(subset=['MARE',metric_name])

            if len(sub) < 2:
                continue

            x = sub['MARE'].values
            y = sub[metric_name].values
            print(metric_name, family, x, y)

            a, b = np.polyfit(x, y, 1)

            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = a*x_line + b

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
                fontsize=11,
                color=family_colors[family]
            )

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

        if c == 0:
            ax.set_ylabel(metric_label, fontsize=14)

# -----------------------------
# Títulos das colunas
# -----------------------------
for c, prob in enumerate(probs):
    axes[0,c].set_title(f"Problema {prob_names[prob]}", fontsize=18)

# -----------------------------
# Rótulos do eixo x
# -----------------------------
for c in range(cols):
    axes[-1,c].set_xlabel("MARE de Validação", fontsize=14)

# -----------------------------
# Legenda
# -----------------------------
color_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]

fig.legend(
    handles=color_handles,
    title="Modelo",
    fontsize=14,
    title_fontsize=14,
    loc="center left",
    bbox_to_anchor=(0.98,0.5)
)

plt.subplots_adjust(wspace=0.05, hspace=0.12)

plt.tight_layout(rect=[0,0,0.95,1])

plt.savefig(
    "Results/plots/UQ_SA_family_linear_scatter.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)