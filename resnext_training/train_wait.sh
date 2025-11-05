#!/bin/bash

# Function to check GPU memory usage
check_gpu_memory() {
    # Get GPU memory usage using nvidia-smi
    local memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    
    # Calculate percentage
    local percentage=$((memory_usage * 100 / memory_total))
    
    echo $percentage
}

# Main loop
while true; do
    usage=$(check_gpu_memory)
    
    if [ "$usage" -lt 10 ]; then
        #echo "Hi - GPU memory usage is ${usage}% (less than 10%)"
	# conda activate videomae2
	python3 dash_resnet.py
	python3 side_resnet.py
	python3 rear_resnet.py
    # echo "Hi"
    break

    else
        TZ='Asia/Kolkata' date
        echo "GPU memory usage is ${usage}% (not less than 10%)"
    fi
    
    # Wait for 5 seconds before checking again
    sleep 60
done
