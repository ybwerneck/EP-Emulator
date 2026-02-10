import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.neural_networks import NModel 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


if len(sys.argv) != 3:
    print("Usage: python load_model.py <SET> <PROB>")
    sys.exit(1)

# Parse arguments from the bash command line
data_set = sys.argv[1]
prob = sys.argv[2]
sets=[data_set]

# Define shared parameters
learning_rate = 0.01
epochs = 1000000

# Define architectures for small, medium, and large networks
architectures = {
    "S": [(nn.SiLU, 16)],  # 2 layers, 128 neurons each
    "M": [(nn.SiLU, 32)] * 2,               # 4 layers, 256 neurons each
    "L": [(nn.SiLU, 64)] * 4             # 8 layers, 512 neurons each
}

# Iterate over each dataset set
for data_set in sets:
    # Load datasets using the current set
    x_train = pd.read_csv(f'data/Generated_Data_{data_set}/Model{prob}/X.csv').values
    y_train = pd.read_csv(f'data/Generated_Data_{data_set}/Model{prob}/Y.csv').values[:,0:]

    # Define dataset specific parameters
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    

    # Iterate over each architecture configuration
    for name, hidden_layer_sizes in architectures.items():
        print(f"Training {name} network on dataset {data_set}...")
        model = NModel(input_size, output_size, hidden_layer_sizes, learning_rate, epochs,name=f"NN_{name}")
        model.train(x_train, y_train, epochs=-1, patience=100,tol=1e-5)
        model.save(f'trainned_models/prob_{prob}/nmodel_{name}_{data_set}.pth')
        print(f"{name.capitalize()} network trained and saved for dataset {data_set}.")

        # Load the model
        model = NModel.load(f'trainned_models/prob_{prob}/nmodel_{name}_{data_set}.pth')
        print(f"{name.capitalize()} model loaded successfully for dataset {data_set}!")

        # Predict and compare
        ypred = model.predict(x_train)
        yval = y_train

        # Plot predictions vs actual values for each output dimension
        fig, axes = plt.subplots(output_size, 1, figsize=(10, 15))
        for i in range(output_size):
            axes[i].scatter(ypred[:, i], yval[:, i], label=f'Feature {i + 1}', alpha=0.6)
            axes[i].set_title(f'{name.capitalize()} Network - Feature {i + 1}')
            axes[i].grid(True)
            axes[i].set_xlim(np.min(ypred[:, i]), np.max(ypred[:, i]))
            axes[i].set_ylim(np.min(yval[:, i]), np.max(yval[:, i]))
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].legend()

        plt.tight_layout()
        os.makedirs(f"Results/Model_fits/{data_set}/prob_{prob}", exist_ok=True)
        plt.savefig(f"Results/Model_fits/{data_set}/prob_{prob}/nmodel_{name}.png")
        plt.close()  # Close the figure after saving to free memory
        print(f"Plot saved as 'results_{data_set}_nmodel_{name}.png'")
