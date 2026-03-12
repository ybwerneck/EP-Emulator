#!/bin/bash

cd EP-Emulator/

conda activate fenics-ompi

SCRIPT=src/inverse_problem/surrogate_inverse.py

echo "Running Model A experiments"
# -------------------------
# Neural Network
# -------------------------
python $SCRIPT \
--model_path trainned_models/prob_A/nmodel_L_1K.pth \
--model_name NN \
--model A


python $SCRIPT \
--model_path trainned_models/prob_A/gp_S_1K.pkl \
--model_name GP \
--model A

# -------------------------
# Polynomial Chaos
# -------------------------
python $SCRIPT \
--model_path trainned_models/prob_A/pce_model2_1K.pth \
--model_name PCE \
--model A

echo "All inverse runs completed"