#!/system/bin/sh

# The following command overclock the Snapdragon 845 CPU & GPU

# CPU
stop mpdecision
for core in `seq 0 7`
do
  echo "Enable performance mode for CPU core $core"
  echo "performance" > /sys/devices/system/cpu/cpu$core/cpufreq/scaling_governor
  echo 1 > /sys/devices/system/cpu/cpu$core/online
done

# GPU
echo "Enable performance mode for GPU"
echo 0 > /sys/class/kgsl/kgsl-3d0/min_pwrlevel
echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel
