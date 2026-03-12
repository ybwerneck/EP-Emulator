import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Configuração
# -----------------------------
blue_colors = sns.color_palette("Blues", 6)[3:]
green_colors = sns.color_palette("Greens", 3)[:]
reds_colors  = sns.color_palette("Reds", 6)[3:]
custom_palette = blue_colors + green_colors + reds_colors 

# Códigos internos -> nomes exibidos em LaTeX
prob_labels = {
    "A": r"$A_M$",
    "B": r"$B_M$",
    "C": r"$A_E$",
    "D": r"$B_E$"
}

probs = ["A","B","C","D"]
data_dict = {p: pd.read_csv(f"Results/inference_{p}.csv") for p in probs}

model_names = ["NN_S", "NN_M", "NN_L", "gp_S", "gp_M", "gp_L", "PCE_2", "PCE_3", "PCE_5"]
color_mapping = {model: custom_palette[i] for i, model in enumerate(model_names)}

# -----------------------------
# Preparação dos dados das curvas de aprendizado
# -----------------------------
records = []

for prob in probs:
    data = data_dict[prob].copy()
    data['MARE_overall'] = data["MARE"]

    for model in model_names:
        sub = data[data['Model'] == model]

        for N in sorted(sub['Training Samples'].unique()):
            vals = sub[sub['Training Samples'] == N]['MARE_overall'].values

            records.append({
                "Problema": prob,
                "Modelo": model,
                "Amostras de Treino": N,
                "Media": vals.mean(),
                "Desvio": vals.std()
            })

lc_df = pd.DataFrame(records)

# -----------------------------
# Gráfico: um subplot por problema
# -----------------------------
n_probs = len(probs)
fig, axes = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=False)
axes = axes.flatten()
if n_probs == 1:
    axes = [axes]

for ax, prob in zip(axes, probs):
    pdf = lc_df[lc_df['Problema'] == prob]

    for model in model_names:
        mdf = pdf[pdf['Modelo'] == model].sort_values("Amostras de Treino")

        if len(mdf) == 0:
            continue

        N    = mdf['Amostras de Treino'].values
        mean = mdf['Media'].values

        ax.plot(
            N, mean,
            color=color_mapping[model],
            linewidth=2
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(prob_labels[prob], fontsize=18)


# Rótulo compartilhado do eixo y
axes[0].set_ylabel("MARE total", fontsize=18)
axes[2].set_ylabel("MARE total", fontsize=18)
axes[0]
# -----------------------------
# Legenda global
# -----------------------------
legend_handles = [Patch(color=color_mapping[m], label=m) for m in model_names]

fig.legend(
    handles=legend_handles,
    title="Modelo substituto",
    fontsize=18,
    title_fontsize=18,
    loc='center left',
    bbox_to_anchor=(0.98, 0.5)
)

# -----------------------------
# Título e layout
# -----------------------------
fig.suptitle("Curvas de aprendizado nos problemas de benchmark", fontsize=22)

plt.tight_layout()
plt.savefig("Results/plots/learning_curve_mare_multiplot.png", dpi=600, bbox_inches="tight")