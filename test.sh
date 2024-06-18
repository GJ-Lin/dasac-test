#!/bin/bash

# 在后台每 1 秒获取一次 GPU 信息并追加到日志文件
> gpu_info.log
watch -n 1 'ixsmi --query-gpu=temperature.gpu,gpu.power.draw,memory.used,utilization.gpu --format=csv,noheader >> gpu_info.log' &

# 获取 watch 命令的 PID
watch_pid=$!

python test.py --mode batch --log_name test_1.log > /dev/null 2>&1 &
python_pid_1=$!
python test.py --mode batch --log_name test_2.log > /dev/null 2>&1 &
python_pid_2=$!

wait $python_pid_1
wait $python_pid_2

# python test.py --mode batch > /dev/null 2>&1
kill $watch_pid