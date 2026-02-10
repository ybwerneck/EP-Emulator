import os
import time
import csv
import numpy as np
import pandas as pd
from ModelA import TTCellModelExt as modelA
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.EP.ModelA import TTCellModelExt as modelA
from src.EP.ModelC import TTCellModelFull as modelB

import chaospy as cp
import os
import sys

# Parse the sample size argument
def parse_sample_size(arg):
    try:
        value = float(arg)
        if value < 1:
            # Convert fractional values (e.g., 0.1) to the corresponding sample count
            return int(value * 1000)
        else:
            # Interpret integers (e.g., 100) as multipliers for 1,000
            return int(value * 1000)
    except ValueError:
        raise ValueError("Invalid argument. Provide a number representing the sample size (e.g., 100 for 100K or 0.1 for 100).")

# Configuration
ti = 11000
tf = 12000
dt = 0.01
dtS = .01
sample_size =   parse_sample_size(sys.argv[1]) if len(sys.argv) > 1 else 100000  # Default to 100K if no argument provided
 # Default to 100K if no argument provided
low, high = 0, 1

# Function to extract QOIs from model output
def extract_qois(results):
    return [{key: value for key, value in result.items() if key != 'Wf'} for result in results]

# Create directory for saving results
output_dir = f"Generated_Data_{sample_size/1000}K"
os.makedirs(output_dir, exist_ok=True)

print(f"Sample size set to: {sample_size}")
print(f"Output directory: {output_dir}")



ugpu=False if sample_size<1000 else True

# Set size parameters for all models
modelA.setSizeParameters(ti, tf, dt, dtS)
modelB.setSizeParameters(ti, tf, dt, dtS)

#########################
# Process for Model A
#########################
performance_A = []

# Sampling for Model A
start = time.time()
dist_A = modelA.getDist(low=0, high=1)
samplesA = dist_A.sample(sample_size, rule="latin_hypercube")
end = time.time()
sampling_time = end - start
performance_A.append(["Sampling", f"{sampling_time:.4f}"])
print(f"Model A - Sampling: {sampling_time:.4f} seconds")

# Running Model A
start = time.time()
results_A = modelA.run(samplesA.T)
end = time.time()
run_time = end - start
performance_A.append(["Running Model", f"{run_time:.4f}"])
print(f"Model A - Running Model: {run_time:.4f} seconds")

# Extracting QOIs for Model A
start = time.time()
qois_A = extract_qois(results_A)
end = time.time()
qoi_time = end - start
performance_A.append(["Extracting QOIs", f"{qoi_time:.4f}"])
print(f"Model A - Extracting QOIs: {qoi_time:.4f} seconds")

# Saving data for Model A
start = time.time()
modelA_dir = os.path.join(output_dir, "ModelA")
os.makedirs(modelA_dir, exist_ok=True)

# Save inputs and outputs
pd.DataFrame(samplesA.T, columns=[f"Input_{i+1}" for i in range(samplesA.shape[0])]).to_csv(
    os.path.join(modelA_dir, "X.csv"), index=False)
pd.DataFrame(qois_A).to_csv(os.path.join(modelA_dir, "Y.csv"), index=False)
end = time.time()
save_time = end - start
performance_A.append(["Saving Data", f"{save_time:.4f}"])
print(f"Model A - Saving Data: {save_time:.4f} seconds")

# Write performance CSV for Model A
csv_path = os.path.join(modelA_dir, "performance.csv")
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Step", "Time (seconds)"])
    writer.writerows(performance_A)


#########################
# Process for Model B
#########################
performance_B = []

# Sampling for Model B
start = time.time()
dist_B = modelB.getDist(low=0.75, high=1.25)
samplesB = dist_B.sample(sample_size, rule="latin_hypercube")
end = time.time()
sampling_time = end - start
performance_B.append(["Sampling", f"{sampling_time:.4f}"])
print(f"Model B - Sampling: {sampling_time:.4f} seconds")

# Running Model B
start = time.time()
results_B = modelB.run(samplesB.T)
end = time.time()
run_time = end - start
performance_B.append(["Running Model", f"{run_time:.4f}"])
print(f"Model B - Running Model: {run_time:.4f} seconds")

# Extracting QOIs for Model B
start = time.time()
qois_B = extract_qois(results_B)
end = time.time()
qoi_time = end - start
performance_B.append(["Extracting QOIs", f"{qoi_time:.4f}"])
print(f"Model B - Extracting QOIs: {qoi_time:.4f} seconds")

# Saving data for Model B
start = time.time()
modelC_dir = os.path.join(output_dir, "ModelB")
os.makedirs(modelC_dir, exist_ok=True)

pd.DataFrame(samplesB.T, columns=[f"Input_{i+1}" for i in range(samplesB.shape[0])]).to_csv(
    os.path.join(modelC_dir, "X.csv"), index=False)
pd.DataFrame(qois_B).to_csv(os.path.join(modelC_dir, "Y.csv"), index=False)
end = time.time()
save_time = end - start
performance_B.append(["Saving Data", f"{save_time:.4f}"])
print(f"Model B - Saving Data: {save_time:.4f} seconds")

# Write performance CSV for Model B
csv_path = os.path.join(modelC_dir, "performance.csv")
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Step", "Time (seconds)"])
    writer.writerows(performance_B)

print(f"Data saved to directory: {output_dir}")



