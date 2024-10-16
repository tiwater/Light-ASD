#!/bin/bash
# cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
# GPU
echo "performance" | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo "performance" | sudo tee /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo "performance" | sudo tee /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
# NPU
echo "performance" | sudo tee /sys/class/devfreq/fdab0000.npu/governor
# GPU
# echo "performance" | sudo tee /sys/class/devfreq/fb000000.gpu/governor