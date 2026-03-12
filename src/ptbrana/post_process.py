import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the custom color palette with distinct groups
blue_colors = sns.color_palette("Blues", 6)[3:]  # Darker blues
green_colors = sns.color_palette("Greens", 3)[:]  # Default greens
yellow_colors = sns.color_palette("Reds", 6)[3:]  # Default yellows
custom_palette = blue_colors + green_colors + yellow_colors

# List of problems
probs = ["A","A"]
# Initialize global limits for consistent scaling across problems
global_limits = {
    "MARE": {"min": 5e-4, "max": 5e-1},
    "R2": {"min": 0.995, "max": 1.00001},
    "Inference Time (s)": {"min": 1e-3, "max": 10},
    "Training Time (s)": {"min": 1e-2, "max": 1e4},
    "Memory (gpu)": {"min": 1e-1, "max": 1e4},
}

# Metrics to process
metrics = list(global_limits.keys())

# Read and process data for all problems
data_dict = {prob: pd.read_csv(f'Results/validation_results_{prob}.csv') for prob in probs}

for k,data in data_dict.items():
        display_names = ["NN_S", "NN_M", "NN_G", "GP_S", "GP_M", "GP_G", "PC_2", "PC_3", "PC_5"]
        model_to_display = dict(zip(data['Model'].unique(), display_names))
        data['Display Model'] = data['Model'].map(model_to_display)
# Generate figures for each metric
if(True):
 for metric in metrics:
    fig, axes = plt.subplots(len(probs), 1, figsize=(12, 6 * len(probs)), sharex=True)

    for i, prob in enumerate(probs):
        ax = axes[i]
        data = data_dict[prob]
        data['Model'] = data['Model'].astype(str)  # Ensure 'Model' is string

        # Create a simple sequential list of model names for display (e.g., Model 1, Model 2, etc.)
        display_names = ["NN_S", "NN_M", "NN_G", "GP_S", "GP_M", "GP_G", "PC_2", "PC_3", "PC_5"]
        model_to_display = dict(zip(data['Model'].unique(), display_names))
        data['Display Model'] = data['Model'].map(model_to_display)

        # Assign colors based on model groups
        num_models = len(data['Display Model'].unique())
        custom_colors = custom_palette[:num_models]

        # Plot for the specific problem using the display names
        sns.barplot(data=data, x='Display Model', y=metric, palette=custom_colors, ax=ax, linewidth=2)

        # Increase font sizes and line widths
        ax.set_yscale('log' if metric != "R2" else 'linear')
        ax.set_ylim(global_limits[metric]["min"], global_limits[metric]["max"])
        ax.tick_params(axis='y', labelsize=20)

        # Set only the last subplot's x-axis label
        if i == len(probs) - 1:
            ax.set_xlabel('Model', fontsize=35)
        else:
            ax.set_xlabel('')

        ax.tick_params(axis='x', rotation=45, labelsize=35)

        # Set ylabel for the first subplot only
        
        ax.set_ylabel(metric, fontsize=30)

    # Add overall title for the figure
    plt.tight_layout()
    plt.subplots_adjust()  # To prevent overlap with the suptitle

    # Save the figure as a high-res PDF
    plt.savefig(f"Results/{metric}_comparison_across_problems.pdf", dpi=300)
    plt.close(fig)
    
    


# Define the custom color palette with distinct groups
blue_colors = sns.color_palette("Blues", 6)[3:]  # Darker blues
green_colors = sns.color_palette("Greens", 3)[:]  # Default greens
yellow_colors = sns.color_palette("Reds", 6)[3:]  # Default yellows
custom_palette = blue_colors + green_colors + yellow_colors
# List of problems
probs = ["A", "B"]
scales = [(1e-3,1)]
# Read the data into a DataFrame (change the path/filename as needed)

for prob in probs:
    df = data_dict[prob]
    # If needed, ensure the 'Display Model' column is a string
    df['Display Model'] = df['Display Model'].astype(str)
    # Define the QoI columns (error metrics)
    qoi_cols = ["MARE_QoI_1", "MARE_QoI_3"]
    # Global limits for each QoI (set as the x-axis limits for the histogram)
    global_limits = {
        "MARE_QoI_0": {"min": 5e-4, "max": 5e-1},
        "MARE_QoI_1": {"min": 5e-4, "max": 5e-1},
        "MARE_QoI_2": {"min": 5e-4, "max": 5e-1},
    }

    num_bins=30

    # Loop through each QoI and create the histogram
    # Assuming you want all histograms in a single figure with stacked subplots vertically
    fig, axes = plt.subplots(len(qoi_cols), 1, figsize=(12, 8 * len(qoi_cols)))  # Create subplots

    # Loop through each qoi and plot in corresponding subplot
    for i, qoi in enumerate(qoi_cols):
        ax = axes[i]  # Access the current subplot

        # Define the bin edges in log space
        bin_edges = np.logspace(np.log10(0.00001), np.log10(0.5), num_bins + 1)
        
        # Create histogram for the current qoi
        sns.histplot(data=df, x=qoi, hue="Display Model", multiple="stack", bins=bin_edges,
                    hue_order=display_names, palette=custom_palette, ax=ax)
        
        # Set the x-axis limits and scale
        ax.set_xlim(1e-5, 0.5)
        ax.set_xscale("log")
        
        ax.tick_params(axis='y', labelleft=False)
             
        # Set ylabel for the first subplot only
        
        ax.set_ylabel("MARE", fontsize=30)
        if(i!=3):
                ax.tick_params(axis='x', rotation=45, labelsize=35)


        # Set labels and title for each subplot
    #  ax.set_xlabel("Error", fontsize=15)
    #  ax.set_ylabel("Count", fontsize=15)
    #ax.set_title(f"Histogram of {qoi}", fontsize=20)
        
        # Format ticks for better readability
        #ax.tick_params(axis="x", rotation=45, labelsize=12)
        #ax.tick_params(axis="y", labelsize=12)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure with all histograms stacked vertically
    plt.savefig(f"Results/all_histograms_stacked{prob}.pdf", dpi=300)
    plt.close(fig)
    
    
    
    
    # Define the training set size column (assumed to be 'Training Set Size')
def convert_set_size(set_str):
        if 'K' in set_str:
            return float(set_str.replace('K', '')) * 1000  # Multiply by 1000 for 'K'
        else:
            return float(set_str)  # Return as is if no 'K'

# Define the number of rows and columns for the grid layout
num_rows = len(qoi_cols)  # One row per QoI
num_cols = len(probs)  # One column per problem

# Create a large figure with subplots arranged in a grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12 * num_cols, 8 * num_rows),sharex="row",sharey="col") 

# Loop through each probability scenario and each QoI to populate the grid
for col, prob in enumerate(probs):
    df = data_dict[prob]
    
    # Ensure the 'Display Model' column is a string
    df['Display Model'] = df['Display Model'].astype(str)
    
    # Apply the conversion to the 'Set' column
    df['Training Set Size'] = df['Set'].apply(convert_set_size)
    df = df.sort_values(by='Training Set Size')
    
    # Loop through each QoI and create a line plot in the corresponding subplot
    for row, qoi in enumerate(qoi_cols):
        ax = axes[row, col]  # Get the subplot for this QoI and this problem
        
        # Loop through each model and plot the line
        i = 0
        for model in display_names:
            # Filter the dataframe for the current model
            model_data = df[df['Display Model'] == model]
            
            # Plot the line: x = training size, y = the current QoI (error metric)
            ax.plot(model_data["Training Set Size"], model_data[qoi], label=model, color=custom_palette[i])
            ax.set_yscale("log")
            i += 1
        ax.tick_params(axis="x", which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis="y", which='both', left=False, right=False, labelleft=False)

        # Set labels and title for the plot
        #ax.set_xlabel("Training Set Size", fontsize=12)
        #ax.set_ylabel(f"{qoi} Error", fontsize=12)
        #ax.set_title(f"{qoi} vs Training Set Size for {prob}", fontsize=14)


        ax.set_ylim(1e-4,1)
        # Add a legend to differentiate models
        ax.legend(title="Models", fontsize=10)
        
        # Format ticks for better readability
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure with all the plots
plt.savefig(f"Results/mega_plot_all_qois.pdf", dpi=300)
plt.close(fig)


metric=metrics[0]
# Define the number of rows and columns for the grid layout
num_rows = len(qoi_cols)  # One row per QoI
num_cols = len(probs)  # One column per problem


ffig, axes = plt.subplots(num_rows, num_cols, figsize=(12 * num_cols, 8 * num_rows), sharex="row", sharey="col")

# Loop through each probability scenario and each QoI to populate the grid
for col, prob in enumerate(probs):
    data = data_dict[prob]
    
    # Ensure 'Model' is a string for proper mapping
    data['Model'] = data['Model'].astype(str)
    
    # Create a simple sequential list of model names for display
    display_names = ["NN_S", "NN_M", "NN_G", "GP_S", "GP_M", "GP_G", "PC_2", "PC_3", "PC_5"]
    model_to_display = dict(zip(data['Model'].unique(), display_names))
    data['Display Model'] = data['Model'].map(model_to_display)

    # Assign colors based on model groups
    custom_colors = sns.color_palette("Blues", 4)

    titles=["APD90","Vreps"]
    
    # Loop through each QoI and create a bar plot in the corresponding subplot
    for row, qoi in enumerate(qoi_cols):
        ax = axes[row, col]  # Get the subplot for this QoI and this problem
        
        # Plot the bar chart for the current problem and QoI
        sns.barplot(data=data, x='Display Model', y=qoi, hue='Training Set Size', 
                    palette=custom_colors, ax=ax, linewidth=10, legend=True, width=0.7)

        ax.legend(fontsize=20)

        # Add a small line above each group to show the trend of increasing training size
        for i, display_model in enumerate(data['Display Model'].unique()):
            group_data = data[data['Display Model'] == display_model]
            training_sizes = group_data['Training Set Size'].values
            qoi_values = group_data[qoi].values
            
            # Fit a line (y = ax + b) to the data
            fit_params = np.polyfit(training_sizes, qoi_values, deg=1)
            a, b = fit_params
            
            print(qoi_values)
            # Get the x-coordinates for the line based on bar positions
            bar_positions = ax.get_xticks()
            x_start = bar_positions[i] - 0.4 # Adjust starting point of the line
            x_end = bar_positions[i] + 0.4    # Adjust ending point of the line
            
            # Generate y-values for the line
            x_vals = np.array([x_start, x_end])
            y_vals = a * training_sizes + b
            
            # Add an offset to place the line above the bars
            y_vals += max(qoi_values) * 0.2
            
            # Plot the line above the bars
            ax.plot([x_start, x_end], [y_vals[0], y_vals[3]], color='red', linestyle='--', linewidth=2)
        
        # Adjust the y-axis scale based on the metric (log scale or linear)
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1)  # Adjust the y-axis limits
        # Rotate x-ticks and set font size
        ax.tick_params(axis='x', rotation=45, labelsize=30)
        ax.set_ylabel('')
        ax.set_xlabel('')

        # Set ylabel for the first column only
        if col != 0:
            ax.tick_params(axis="y", which='both', left=False, right=False, labelleft=False)
        else:
            ax.set_ylabel(titles[row],fontsize=30)

        
        ax.tick_params(axis='y', labelsize=30)

# Adjust layout to avoid overlap
plt.suptitle(f"{metric} Comparison Across Problems", fontsize=25, y=1.02)
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)  # Adjust space between subplots

# Save the figure as a high-res PDF
plt.savefig(f"Results/{metric}_AAcomparison_across_problems.pdf", dpi=300)
plt.close(fig)
print("end")


df=data

# Extract x and y values
x = ["IT_1600", "IT_3200", "IT_6400", "IT_12800", "IT_25600", "IT_51200","IT_102400"]
x_values = [int(i.split("_")[1]) for i in x]

# Plot each model

# Map unique model names to colors
unique_models = df["Model"].unique()
color_map = {model: custom_palette[i % len(custom_palette)] for i, model in enumerate(unique_models)}

# Plot each model with assigned colors
plt.figure(figsize=(10, 6))
added_labels=set()
i=0
for index, row in df.iterrows():
    print(row)
    y_values = row[x].values
    label = unique_models[i] if unique_models[i] not in added_labels else None
    plt.plot(x_values, y_values, marker='o', label=label, color=color_map[row["Model"]], alpha=0.5)
    added_labels.add(unique_models[i])
    i=i+1
    if(i==len(unique_models)):
        i=0
# Customize the plot
plt.title("Model Performance Comparison")
plt.xlabel("N of samples")
plt.ylabel("Y (Metric Values)")
plt.legend(title="Models",loc="lower right", fontsize=10)
#plt.grid(True)

plt.yscale('log')
plt.xscale('log')
plt.savefig("n_study.pdf", dpi=300)