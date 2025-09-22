import numpy as np
import pandas as pd
import pickle
import time  # Import the time module



# Define model paths
print("Beginning inference stage")
probs = ["A","B"]
sets = ["0.1K","0.5K","1K","5K"]  # List of sets to process


for prob in probs:
    model_paths = []
    for setm in sets:
        paths = [
            (f'models/prob_{prob}/nmodel_small_{setm}.pth', setm),
            (f'models/prob_{prob}/nmodel_medium_{setm}.pth', setm),
            (f'models/prob_{prob}/nmodel_large_{setm}.pth', setm),
            (f'models/prob_{prob}/gp_Small_{setm}.pkl', setm),
            (f'models/prob_{prob}/gp_Medium_{setm}.pkl', setm),
            (f'models/prob_{prob}/gp_Large_{setm}.pkl', setm),
      #      (f'models/prob_{prob}/gp_skt_Small_{setm}.pkl', setm),
      #      (f'models/prob_{prob}/gp_skt_Medium_{setm}.pkl', setm),
      #      (f'models/prob_{prob}/gp_skt_Large_{setm}.pkl', setm),
            (f'models/prob_{prob}/pce_model2_{setm}.pth', setm),
            (f'models/prob_{prob}/pce_model3_{setm}.pth', setm),
            (f'models/prob_{prob}/pce_model5_{setm}.pth', setm),
        ]
        model_paths.extend(paths)  # Add the paths and their corresponding sets

    # Print all assembled paths for verification
    for path, setm in model_paths:
        print(f"Path: {path}, Set: {setm}")

    # Load models
    models = []
    for path, setm in model_paths:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            models.append((model, setm, path))  # Store the model with its associated set and path

    print("Models loaded successfully.")

    # Load validation dataset
    x_val = pd.read_csv(f'Generated_Data_100K/Model{prob}/X.csv').values
    y_val = pd.read_csv(f'Generated_Data_100K/Model{prob}/Y.csv').values

    # Define metrics
    metrics = {
        "MARE": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred) / np.abs(y_true)),
        "MSE": lambda y_true, y_pred: np.mean(((y_true - y_pred) / np.abs(y_true)) ** 2),
        "R2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }

    for i in range(4):  # Assuming there are 4 QoIs (indexed 0 to 3)
        metrics[f"MARE_QoI_{i}"] = lambda y_true, y_pred, i=i: np.mean(
            np.abs(y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i])
        )
        metrics[f"MSE_QoI_{i}"] = lambda y_true, y_pred, i=i: np.mean(
            ((y_true[:, i] - y_pred[:, i]) / np.abs(y_true[:, i])) ** 2
        )
    # Validate models and store results
    results = []
    k = 0
    for model, setm, path in models:
        model_name = path.split('/')[-1].split('_')[:-1]  # Extract model name from the path
        print(model_name)
        for _ in range(10):
            y_pred, elapsed_time = model.predict(x_val, meas_time=True)
        num_runs = 10
        all_times = []

        for _ in range(num_runs):
            y_pred, elapsed_time = model.predict(x_val, meas_time=True)
            all_times.append(elapsed_time)
            print(f"Run {_ + 1}: Elapsed Time: {elapsed_time:.4f}s")

        elapsed_time = np.mean(all_times)
        print(y_pred, elapsed_time)
        # Save the emulator (model) to disk
        # Save the emulator (model) to disk

        # Prepare DataFrame with headers separating inputs and outputs, including true outputs
        input_cols = [f"input_{i}" for i in range(x_val.shape[1])]
        true_output_cols = [f"true_output_{i}" for i in range(y_val.shape[1])]
        pred_output_cols = [f"pred_output_{i}" for i in range(y_pred.shape[1])]

        pred_df = pd.DataFrame(
            np.column_stack([x_val, y_val, y_pred]),
            columns=input_cols + true_output_cols + pred_output_cols
        )
        print(pred_df)
        pred_df.to_csv(
            f'Results/{setm}/prob_{prob}/predictions_{"_".join(model_name)}.csv',
            index=False
        )

        
        model_results = {"Model": model_name}

        for metric_name, metric_fn in metrics.items():
            model_results[metric_name] = metric_fn(y_val, y_pred)
        model_results["Inference Time (s)"] = elapsed_time  # Store the timing
        model_results["Training Time (s)"] = model.metadata.get("time_train", "-1")
        model_results["Memory (gpu)"] = model.metadata.get("gpu_memory_MB", "-1")
        model_results["Set"] = setm  # Add the set information
        model_results["id"] = k
        model_results["traing_it"] = model.metadata.get("training_epochs", "-1")


        subset_sizes= [100*(2**i) for i in range (0,12)] 
        # Define subset sizes to evaluate
        for subset_size in subset_sizes:
            print(f"Evaluating subset size: {subset_size}")
            # Sample a subset of the validation data
            indices = np.random.choice(len(x_val), size=min(subset_size, len(x_val)), replace=False)
            x_subset = x_val[indices]
            y_subset = y_val[indices]

            # Measure inference time for the subset
            num_runs = 10
            all_times = []
            for _ in range(num_runs):
                y_pred, elapsed_time = model.predict(x_subset, meas_time=True)
                all_times.append(elapsed_time)
                print(f"Run {_ + 1}, Subset {subset_size}: Elapsed Time: {elapsed_time:.4f}s")

            avg_elapsed_time = np.mean(all_times)
            print(f"Average Elapsed Time for Subset {subset_size}: {avg_elapsed_time:.4f}s")

            # Calculate metrics for the subset
            model_results[f"IT_{subset_size}"] = avg_elapsed_time  # Store the timing


        k += 1
        results.append(model_results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Force the DataFrame to use scientific notation with 3 decimal places for float values

    # Print results
    print("Validation Results:")
    print(results_df)

    # Save results to CSV with scientific notation and 3 decimals
    results_df.to_csv(f'Results/validation_results_{prob}.csv', index=False)
    print(f"Validation results saved to 'validation_results_{prob}.csv'.")

