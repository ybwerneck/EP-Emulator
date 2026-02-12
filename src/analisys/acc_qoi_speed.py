import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Define the custom color palette with distinct groups
blue_colors = sns.color_palette("Blues", 6)[3:]  # Darker blues
green_colors = sns.color_palette("Greens", 3)[:]  # Default greens
reds_colors = sns.color_palette("Reds", 6)[3:]  # Default reds
black_colors = sns.color_palette("dark:gray", 3)  # Black/Gray tones
custom_palette = blue_colors + green_colors + reds_colors + black_colors


# List of problems
probs = ["B"]


# Initialize global limits for consistent scaling across problems
global_limits = {
    "MSE": {"min": 5e-6, "max": 1e-2, "name":"MSE"},
    "R2": {"min": 0.995, "max": 1.00001, "name":"R2"},
    "Inference Time (s)": {"min": 1e-3, "max": 5e2, "name":"Tempo de Inferência (s)"},
    "Training Time (s)": {"min": 1e-1, "max": 5e4, "name":"Tempo de Treino (s)"},
    "Memory (gpu)": {"min": 1e1, "max": 2e3, "name":"Memória GPU (MB)"},
}


# Read and process data for all problems
data_dict = {prob: pd.read_csv(f'Results/inference_{prob}.csv') for prob in probs}


# Process data
for prob in probs:
    data = data_dict[prob]
    data['Model'] = data['Model'].astype(str)
    
    # Create display names
    display_names = ["NN_P", "NN_M", "NN_G", "GP_P", "GP_M", "GP_G", 
                     "PC_2", "PC_3", "PC_5"]
    model_to_display = dict(zip(data['Model'].unique(), display_names))
    data['Display Model'] = data['Model'].map(model_to_display)


    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Output names for each QoI
    output_names = {0: 'dVmax', 1: 'APD90', 2: 'APD50', 3: 'Vreps'}
    
    for i in range(4):
        ax = axes[i]
        output_name = output_names[i]
        
        # Calcular o valor máximo do MSE para a variável correspondente
        max_mse = data[f'MARE_QoI_{i}'].max()
        
        # Normalizar os valores de MSE dividindo pelo valor máximo
        #data[f'MARE_QoI_{i}'] = data[f'MSE_QoI_{i}'] / max_mse
        
        # Create scatter plot - manter legenda apenas no primeiro subplot
        show_legend = (i == 0)
        scatter = sns.scatterplot(data=data, x='Inference Time (s)', y=f'MARE_QoI_{i}', 
                                 hue='Display Model', palette=custom_palette, s=100, 
                                 edgecolor='black', ax=ax, legend=show_legend)
        
        # Set scales
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Set title for each subplot
        ax.set_title(f'QoI {output_name}', fontsize=16, fontweight='bold')
        
        # Set labels only for appropriate positions
        # Y-label only for first column (i=0,2)
        if i % 2 == 0:  # First column
            ax.set_ylabel('MARE', fontsize=14)
        else:
            ax.set_ylabel('')
        
        # X-label only for bottom row (i=2,3)
        if i >= 2:  # Bottom row
            ax.set_xlabel('Inference time (s)', fontsize=14)
        else:
            ax.set_xlabel('')
    
    # Mover a legenda do primeiro subplot para fora da figura
    legend = axes[0].get_legend()
    if legend:
        # Remove a legenda do primeiro subplot
        legend.remove()
        
        # Criar uma nova legenda para toda a figura
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Emulador', fontsize=12, title_fontsize=12, 
                   loc='center right', bbox_to_anchor=(0.999, 0.5))
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    
    # Save the figure
    plt.savefig(f'Results/plots/mse_inference_all_QoIs{prob}.png', 
                dpi=600, bbox_inches='tight')
    # plt.show()