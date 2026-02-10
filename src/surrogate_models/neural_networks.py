
import torch
import torch.nn as nn
import torch.optim as optim
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.DD_Models import ModelInterface 



##Underlying Fully Connected Network
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_sizes, dtype=torch.float32, device='cpu'):
        """
        Initialize the Fully Connected Network.

        Args:
            input_shape (int): Number of input features.
            output_shape (int): Number of output features.
            hidden_sizes (list): Hidden layer sizes and optionally activation functions.
            dtype (torch.dtype): Data type for the model.
            device (str or torch.device): Device to run the model on ('cpu' or 'cuda').
        """
        super(FullyConnectedNetwork, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.layers = nn.ModuleList()

        self.init_layers(input_shape, output_shape, hidden_sizes)
        self.to(self.device)  # Move model to the specified device

    def init_layers(self, input_shape, output_shape, hidden_sizes):
        """
        Initialize the layers depending on the structure of `hidden_sizes`.

        - If `hidden_sizes` is a list of integers, use fixed activation (e.g., Tanh).
        - If `hidden_sizes` is a list of tuples (activation, size), use the specified activation.
        """
        in_features = input_shape

        if all(isinstance(h, int) for h in hidden_sizes):
            # Case: List of integers with fixed activation (Tanh)
            for hidden_size in hidden_sizes:
                self.layers.append(nn.Linear(in_features, hidden_size, bias=True, dtype=self.dtype))
                self.layers.append(nn.Tanh())
                in_features = hidden_size
        elif all(isinstance(h, tuple) and len(h) == 2 for h in hidden_sizes):
            # Case: List of tuples with activation and size
            for act, hidden_size in hidden_sizes:
                self.layers.append(nn.Linear(in_features, hidden_size, bias=True, dtype=self.dtype))
                self.layers.append(act())
                in_features = hidden_size
        else:
            raise ValueError("hidden_sizes must be a list of integers or a list of (activation, size) tuples.")

        # Add the output layer
        self.layers.append(nn.Linear(in_features, output_shape, bias=True, dtype=self.dtype))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the network.
        """
        x = x.to(self.device)  # Ensure input is on the correct device
        for layer in self.layers:
            x = layer(x)
        return x


##Neural Network Model Wrapper
class NModel(ModelInterface):
    def __init__(self, input_size, output_size, hidden_layer_sizes=(20,), learning_rate=0.01, epochs=1000, device=None, name=None):
        super().__init__(name=name)
        self.type="Neural_Network"
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs  # Maximum epochs. Use -1 for no maximum.
        self.device = device or (torch.device("cuda"))
        
        # Neural network model
        self.model = FullyConnectedNetwork(input_size, output_size, hidden_layer_sizes, device=self.device)
        self.loss_function = nn.L1Loss()  # Using MAE as training metric
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Metadata
        self.metadata.update({
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate": learning_rate,
            "epochs": epochs,
        })


    def _train(self, x, y, **kwargs):
        tol = kwargs.get("tol", 1e-5)
        patience = kwargs.get("patience", 10)
        best_mae = float("inf")
        epochs_without_improvement = 0

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.9,     # reduce LR by half
            patience=patience//10,     # wait 5 epochs without improvement
            threshold=tol,  # same threshold as tol
        )

        # Determine maximum epochs: if self.epochs is -1, run indefinitely.
        max_epochs = self.epochs if self.epochs != -1 else float('inf')
        epoch = 0

        while epoch < max_epochs:
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(x_tensor)
            # Compute MAE on the training set
            train_mae = self.loss_function(predictions, y_tensor)
            train_mae.backward()
            self.optimizer.step()

            # Step the scheduler (use MAE as metric)
            scheduler.step(train_mae.item())

            if epoch % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Training MAE: {train_mae.item():.4f}, LR: {current_lr:.6f}")

            # Early stopping based on training MAE improvement
            if train_mae.item() < best_mae - tol:
                best_mae = train_mae.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} with training MAE {train_mae.item():.4f}")
                break

            epoch += 1

        self.metadata["training_epochs"] = epoch


    def _predict(self, X, **kwargs):
        ##this shoud be used for high level predictions without gradient and cpu called
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(X_tensor).to("cpu").numpy()


    ##this is fully diferentiable
    def _forward(self, X, **kwargs):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model(X)