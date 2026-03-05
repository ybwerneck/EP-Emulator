# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Define the custom color palette with distinct groups
# blue_colors = sns.color_palette("Blues", 6)[3:]  # Darker blues
# green_colors = sns.color_palette("Greens", 3)[:]  # Default greens
# reds_colors = sns.color_palette("Reds", 6)[3:]  # Default reds
# black_colors = sns.color_palette("dark:gray", 3)  # Black/Gray tones
# custom_palette = blue_colors + green_colors + reds_colors + black_colors

# # List of problems
# probs = ["B"]

# # Initialize global limits for consistent scaling across problems
# global_limits = {
#     # "MARE": {"min": 5e-4, "max": 5e-1},
#     "MSE": {"min": 5e-6, "max": 1e-2, "name":"MSE"},
#     "R2": {"min": 0.995, "max": 1.00001, "name":"R2"},
#     "Inference Time (s)": {"min": 1e-3, "max": 5e2, "name":"Tempo de Inferência (s)"},
#     "Training Time (s)": {"min": 1e-1, "max": 5e4, "name":"Tempo de Treino (s)"},
#     "Memory (gpu)": {"min": 1e1, "max": 2e3, "name":"Memória GPU (MB)"},
# }

# # Metrics to process
# metrics = list(global_limits.keys())

# # Read and process data for all problems
# data_dict = {prob: pd.read_csv(f'Results/validation_new_results_{prob}.csv') for prob in probs}

# # Generate figures for each metric
# for metric in metrics:
#     # fig, axes = plt.subplots(len(probs), 1, figsize=(12, 8 * len(probs)), sharex=True)

#     for i, prob in enumerate(probs):
#         # ax = axes
#         data = data_dict[prob]
#         data['Model'] = data['Model'].astype(str)  # Ensure 'Model' is string

#         # Create a simple sequential list of model names for display (e.g., Model 1, Model 2, etc.)
#         display_names = ["NN_S", "NN_M", "NN_G", "GP_S", "GP_M", "GP_G", "GP_SK_S", "GP_SK_M", "GP_SK_G", "PC_2", "PC_3", "PC_5"]
#         # display_names = ["NN_S", "NN_M", "NN_G", "GP_S", "GP_M", "GP_G"]
#         model_to_display = dict(zip(data['Model'].unique(), display_names))
#         data['Display Model'] = data['Model'].map(model_to_display)

#         # Assign colors based on model groups
#         num_models = len(data['Display Model'].unique())
#         custom_colors = custom_palette[:num_models]

#         # Plot for the specific problem using the display names
#     #     sns.barplot(data=data, x='Display Model', y=metric, palette=custom_colors, ax=ax, linewidth=2)

#     #     # Increase font sizes and line widths
#     #     ax.set_yscale('log' if metric != "R2" else 'linear')
#     #     ax.set_ylim(global_limits[metric]["min"], global_limits[metric]["max"])
#     #     ax.tick_params(axis='y', labelsize=20)

#     #     # Set only the last subplot's x-axis label
#     #     if i == len(probs) - 1:
#     #         ax.set_xlabel('Emulador', fontsize=25)
#     #     else:
#     #         ax.set_xlabel('')

#     #     ax.tick_params(axis='x', rotation=45, labelsize=25)

#     #     # Set ylabel for the first subplot only
        
#     #     ax.set_ylabel(global_limits[metric]["name"], fontsize=20)

#     # # Add overall title for the figure
#     # plt.tight_layout()
#     # plt.subplots_adjust()  # To prevent overlap with the suptitle
#     # plt.show()

#     # Save the figure as a high-res PDF
#     # plt.savefig(f"Results/{metric}_single.pdf", dpi=300)
#     # plt.close(fig)



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# for i in np.arange(0,4) :
#     output_names = {0: 'dVmax', 1: 'APD90', 2: 'APD50', 3: 'Vreps'}
#     output_name = output_names.get(i, f'Output_{i}')
#     # Generate the scatter plot: MAE vs Inference Time
#     # Calcular o valor máximo do MSE para a variável correspondente
#     max_mse = data[f'MSE_QoI_{i}'].max()
    
#     # Normalizar os valores de MSE dividindo pelo valor máximo
#     data[f'Normalized_MSE_QoI_{i}'] = data[f'MSE_QoI_{i}'] / max_mse
    
#     # Gerar o gráfico de dispersão: MSE normalizado vs Tempo de Inferência
#     plt.figure(figsize=(10, 6))

#     sns.scatterplot(data=data, x='Inference Time (s)', y=f'Normalized_MSE_QoI_{i}', hue='Display Model', palette=custom_palette, s=100, edgecolor='black')

#     # Melhorar a aparência do gráfico
#     plt.xlabel('Tempo de Inferência (s)', fontsize=22)
#     plt.ylabel(f'MSE de {output_name} (Normalizado)', fontsize=22)
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     plt.legend(title='Emulador', fontsize=14, loc="best")

#     # Display the plot
#     plt.tight_layout()
#     plt.show()
# #     plt.savefig(f'C:/Users/Lucas Teixeira/OneDrive/Mestrado/Dissertação/Comparativo Emuladores/Model_{prob}/QoIs/mse_inference_QoI_{output_name}.png', dpi=600)





































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
data_dict = {prob: pd.read_csv(f'Results/validation_new_results_{prob}.csv') for prob in probs}


# Process data
for prob in probs:
    data = data_dict[prob]
    data['Model'] = data['Model'].astype(str)
    
    # Create display names
    display_names = ["NN_P", "NN_M", "NN_G", "GP_P", "GP_M", "GP_G", 
                     "GP_SK_P", "GP_SK_M", "GP_SK_G", "PC_2", "PC_3", "PC_5"]
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
    plt.savefig(f'mse_inference_all_QoIs{prob}.png', 
                dpi=600, bbox_inches='tight')
    # plt.show()