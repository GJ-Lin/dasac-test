#!/bin/bash

> gpu_info.log
watch -n 1 'ixsmi --query-gpu=temperature.gpu,gpu.power.draw,memory.used,utilization.gpu --format=csv,noheader >> gpu_info.log' &
# watch -n 1 'nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader >> gpu_info.log' &
watch_pid=$!

python test.py --mode batch > /dev/null 2>&1
kill $watch_pid