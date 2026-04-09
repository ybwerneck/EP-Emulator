#!/bin/bash

# Check parameters
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <IP> <PROB> <SET>"
    exit 1
fi

IP=$1
PROB=$2
SET=$3

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Local log files
LOG_NN="logs/run_${PROB}_${IP}_${SET}_NN.out"
LOG_GAUSS="logs/run_${PROB}_${IP}_${SET}_GP.out"
LOG_PCE="logs/run_${PROB}_${IP}_${SET}_PCE.out"

# Function to check SSH connectivity
check_ssh() {
    sshpass -p "991215" ssh -o BatchMode=yes -o ConnectTimeout=5 yan@$IP "echo 1" > "logs/run_${PROB}_${IP}_${SET}.out" 2>&1 &
    if [ $? -ne 0 ]; then
        echo "ERROR: Cannot connect to $IP via SSH with sshpass. Check IP, username, or password."
        exit 1
    fi
}

# Function to run remote command and log output
run_remote() {
    local CMD="$1"
    local LOG_FILE="$2"

    sshpass -p "991215" ssh -t yan@$IP bash -c "'
        echo strat
        
        conda activate torchcuda
        $CMD
        echo done
    '" 2>&1 | tee "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Remote command failed. Check log $LOG_FILE"
        exit 1
    fi
}

# 0️⃣ Check SSH connection first
check_ssh

# 1️⃣ Train NN
run_remote "echo 'Startsing NN training for set $SET'; python train_NN.py $SET $PROB; echo 'Finished NN training'" "$LOG_NN"

# 2️⃣ Train Gaussian
run_remote "echo 'Startsing Gaussian training for set $SET'; python train_gaussian.py $SET $PROB; echo 'Finished Gaussian training'" "$LOG_GAUSS"

# 3️⃣ Train PCE
run_remote "echo 'Startsing PCE training for set $SET'; python train_pce.py $SET $PROB; echo 'Finished PCE training'" "$LOG_PCE"

echo "All simulations executed on $IP with set: $SET and problem: $PROB"
echo "Logs saved in $LOG_DIR/"
