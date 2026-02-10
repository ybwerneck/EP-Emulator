import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.surrogate_models.pce import PCEModel  # Ensure this matches the location of your NModel class
import chaospy as cp

# Define the dataset sets to work with
import sys
if len(sys.argv) != 3:
    print("Usage: python load_model.py <SET> <PROB>")
    sys.exit(1)

# Parse arguments from the bash command line
data_set = sys.argv[1]
prb = sys.argv[2]
sets=[data_set]

Ps= [2,3,5]  # Polynomial order

# Iterate over each dataset set
for P in Ps:
 for data_set in sets:
    # Load datasets using the current set identifier
    x_train = pd.read_csv(f'data/Generated_Data_{data_set}/Model{prb}/X.csv').values
    y_train = pd.read_csv(f'data/Generated_Data_{data_set}/Model{prb}/Y.csv').values
    print(f"Loaded data for dataset {data_set} with input shape {x_train.shape} and output shape {y_train.shape}")
    # Define PCE parameters using the training data
    dist = cp.J(*[cp.Uniform(0, 1) for i in range(x_train.shape[1])])

    # Create and train the PCE model
    print(f"Training PCE model for dataset {data_set}...")
    pce_model = PCEModel(dist, P,name=f"PCE_{P}")
    pce_model.train(x_train, y_train)
    pce_model.save(f'trainned_models/prob_{prb}/pce_model{P}_{data_set}.pth')
    print(f"PCE model trained and saved for dataset {data_set}.")

    # Load the model
    pce_model = PCEModel.load(f'trainned_models/prob_{prb}/pce_model{P}_{data_set}.pth')
    print(f"PCE model loaded successfully for dataset {data_set}!")

    # Predict and compare
    ypred = pce_model.predict(x_train)
    yval = y_train

    # Plot predictions vs actual values for each output dimension
    output_size = y_train.shape[1]
    fig, axes = plt.subplots(output_size, 1, figsize=(10, 15))
    print(np.shape(ypred[:, 0]))
    for i in range(output_size):
        axes[i].scatter(ypred[:, i], yval[:, i], label=f'Feature {i + 1}', alpha=0.6)
        axes[i].set_title(f'PCE Model - Feature {i + 1}')
        axes[i].grid(True)
        axes[i].set_xlim(np.min(ypred[:, i]), np.max(ypred[:, i]))
        axes[i].set_ylim(np.min(yval[:, i]), np.max(yval[:, i]))
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].legend()

    plt.tight_layout()
    os.makedirs(f"Results/Model_fits/{data_set}/prob_{prb}", exist_ok=True)
    plt.savefig(f"Results/Model_fits/{data_set}/prob_{prb}/pce_model{P}.png")
    plt.close()  # Close the plot to free memory
    print(f"Plot saved as '{data_set}_results_pce_model{P}.png'")
