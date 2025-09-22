#!/bin/bash

IPS=( "10.22.10.119" )
PROBS=( "A" "B" )
SETS=( "0.1K" "0.5K" "1K" "5K" )

mkdir -p logs

# Create a temporary directory for machine locks
LOCK_DIR=$(mktemp -d)
for IP in "${IPS[@]}"; do
    touch "$LOCK_DIR/$IP.lock"
done

# Build job list
JOBS=()
for SET in "${SETS[@]}"; do
    for i in "${!PROBS[@]}"; do
        PROB="${PROBS[$i]}"
        JOBS+=("$PROB $SET")
    done
done

job_index=0
num_jobs=${#JOBS[@]}

while [ $job_index -lt $num_jobs ]; do
    for IP in "${IPS[@]}"; do
        # If machine is free
        if [ ! -f "$LOCK_DIR/$IP.lock.running" ]; then
            read PROB SET <<< "${JOBS[$job_index]}"
            mkdir -p "Results/$SET/prob_$PROB"

            echo "Dispatching job $job_index -> Machine $IP (PROB=$PROB, SET=$SET)"

            # Mark machine as running
            touch "$LOCK_DIR/$IP.lock.running"

            # Launch job in background
            bash bash/sequential_local_trainning.bash "$IP" "$PROB" "$SET" > "logs/run_${PROB}_${IP}_${SET}.out" 2>&1 &

            # After job finishes, remove the running lock
            PID=$!
            ( wait $PID; rm -f "$LOCK_DIR/$IP.lock.running" ) &

            ((job_index++))
            # Break if no more jobs
            if [ $job_index -ge $num_jobs ]; then
                break
            fi
        fi
    done
    sleep 5  # small delay to avoid busy-waiting
done

# Wait for all background jobs to finish
wait
rm -rf "$LOCK_DIR"

echo "All jobs finished."
