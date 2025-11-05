#!/bin/bash

LOGFILE="gpu_usage_log_resnet_224.csv"

# Write header
echo "timestamp,index,name,gpu_utilization[%],memory.total[MiB],memory.used[MiB],memory.free[MiB],temperature[°C]" > "$LOGFILE"

# Loop and log every 5 seconds
while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits >> "$LOGFILE"
    sleep 1
done
