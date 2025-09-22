import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# --- 0. Carregar os dados ---

prb = "B"
X = pd.read_csv(f'Generated_Data_1.0K/Model{prb}/X.csv')
y = pd.read_csv(f'Generated_Data_1.0K/Model{prb}/Y.csv')

# --- 1. Pré-processamento ---
# Remover tdV
y = y.drop(columns=['tdV'])

# Renomear colunas de entrada (X)
X.columns = [
'hypo','hyper','acid','gna','gcal','gk1','gkr','gks','gto','gbca','gpk','g_pc']


# Amostragem (20%)
sample_size = int(0.2 * len(X))
X_sample = X.sample(n=sample_size, random_state=42)
y_sample = y.sample(n=sample_size, random_state=42)

print(X_sample.head())
print(y_sample.head())

# --- 2. Distribuições marginais ---

n_inputs = X.shape[1]

# Correlação entre entradas e QoIs
corr_matrix = pd.concat([X, y_sample], axis=1).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix.iloc[:len(X.columns), len(X.columns):], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlação entre entradas e QoIs')
plt.savefig("qoicor.png")

# --- 4. Matriz de Dispersão (pairplot) das entradas ---

sns.pairplot(X_sample)  # mostra só primeiras 6 variáveis para facilitar visualização
plt.suptitle('Matriz de Dispersão das Variáveis de Entrada (parcial - amostragem)', y=1.02)
plt.savefig("disp.png")

# --- 5. Mapeamento X → y (scatterplots) ---

# Primeiro gráfico - 1 à 3
fig, axes = plt.subplots(4, 3, figsize=(10, 12))  # 4 QoIs x 3 primeiras variáveis
for i, output in enumerate(y.columns):
    for j in range(3):  # Apenas Hipóxia, Hipercalemia e Acidose nos scatterplots iniciais
        ax = axes[i, j]
        ax.scatter(X_sample.iloc[:, j], y_sample[output], alpha=0.6, c='purple', edgecolors='k', s=10)
        ax.set_xlabel(X_sample.columns[j])
        ax.set_ylabel(output)
plt.tight_layout()
plt.savefig("qoicor1.png")

# Segundo gráfico - 4 à 6
fig, axes = plt.subplots(4, 3, figsize=(10, 12))  # 4 QoIs x 3 primeiras variáveis
for i, output in enumerate(y.columns):
    for j in range(3):  # Apenas Hipóxia, Hipercalemia e Acidose nos scatterplots iniciais
        ax = axes[i, j]
        j_new = j + 3
        ax.scatter(X_sample.iloc[:, j_new], y_sample[output], alpha=0.6, c='purple', edgecolors='k', s=10)
        ax.set_xlabel(X_sample.columns[j_new])
        ax.set_ylabel(output)
plt.tight_layout()
plt.savefig("qoicor2.png")

# Terceiro gráfico - 7 à 9
fig, axes = plt.subplots(4, 3, figsize=(10, 12))  # 4 QoIs x 3 primeiras variáveis
for i, output in enumerate(y.columns):
    for j in range(3):  # Apenas Hipóxia, Hipercalemia e Acidose nos scatterplots iniciais
        ax = axes[i, j]
        j_new = j + 6
        ax.scatter(X_sample.iloc[:, j_new], y_sample[output], alpha=0.6, c='purple', edgecolors='k', s=10)
        ax.set_xlabel(X_sample.columns[j_new])
        ax.set_ylabel(output)
plt.tight_layout()
plt.savefig("qoicor3.png")

# Quarto gráfico - 10 à 12
fig, axes = plt.subplots(4, 3, figsize=(10, 12))  # 4 QoIs x 3 primeiras variáveis
for i, output in enumerate(y.columns):
    for j in range(3):  # Apenas Hipóxia, Hipercalemia e Acidose nos scatterplots iniciais
        ax = axes[i, j]
        j_new = j + 9
        ax.scatter(X_sample.iloc[:, j_new], y_sample[output], alpha=0.6, c='purple', edgecolors='k', s=10)
        ax.set_xlabel(X_sample.columns[j_new])
        ax.set_ylabel(output)
plt.tight_layout()
plt.savefig("qoicor4.png")
# --- 6. Escalonamento e Normalização ---

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_sample), columns=X.columns)
y_scaled = pd.DataFrame(scaler.fit_transform(y_sample), columns=y.columns)

# Histogramas de variáveis escaladas (X)
fig, axes = plt.subplots(n_inputs // 4 + 1, 4, figsize=(10, 2.5 * (n_inputs // 4 + 1)))
axes = axes.flatten()
for i, col in enumerate(X_scaled.columns):
    sns.histplot(X_scaled[col], kde=True, ax=axes[i], color='skyblue', bins=30)
    axes[i].set_title(f'Histograma da variável {col} (Normalizado)')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequência')
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig("qoicor5.png")

# Histogramas de QoIs escaladas (y)
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
for i, col in enumerate(y_scaled.columns):
    sns.histplot(y_scaled[col], kde=True, ax=axes[i], color='salmon', bins=30)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequência')
fig.suptitle('Histograma das QoIs Normalizadas', fontsize=14)
plt.tight_layout()
plt.savefig("qoicor6.png")
