#!/bin/bash

> gpu_info.log
watch -n 1 'ixsmi --query-gpu=temperature.gpu,gpu.power.draw,memory.used,utilization.gpu --format=csv,noheader >> gpu_info.log' &
# watch -n 1 'nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader >> gpu_info.log' &
watch_pid=$!

python test.py --mode batch --log_name test_1.log > /dev/null 2>&1 &
python_pid_1=$!
python test.py --mode batch --log_name test_2.log > /dev/null 2>&1 &
python_pid_2=$!

wait $python_pid_1
wait $python_pid_2

kill $watch_pid