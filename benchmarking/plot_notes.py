# TF_Record: 
# Disk size = 8039.7179 MB  ; Mem. size  = 8039.7164 MB;     Num. traj = 51
# Read:  latency = 19.6558 s; throughput = 409.0250 MB/s,    2.5947 traj/s
# Write: latency = 0.0526 s ; throughput = 152931.1876 MB/s, 970.1199 traj/s

"""###"""

# Fog-X:
# Disk size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 42.1416 s ; throughput = 159.1630 MB/s, 1.2102 traj/s
# Write: latency = 258.8635 s; throughput = 25.9109 MB/s,  0.1970 traj/s

# Polars:
# Disk size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 22.9974 s ; throughput = 291.6581 MB/s, 2.2176 traj/s
# Write: latency = 241.4137 s; throughput = 27.7838 MB/s,  0.2113 traj/s

# PyArrow:
# Disk size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 27.9808 s ; throughput = 239.7136 MB/s, 1.8227 traj/s
# Write: latency = 80.9986 s ; throughput = 82.8086 MB/s,  0.6296 traj/s

# Pandas:
# Disk size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 49.1056 s ; throughput = 136.5909 MB/s, 1.0386 traj/s
# Write: latency = 103.6820 s; throughput = 64.6918 MB/s,  0.4919 traj/s

"""###"""

# Arrow_REG:
# Disk size = 14651.9697 MB ; Num. traj = 51
# Read:  latency = 39.5354 s; throughput = 370.6041 MB/s, 1.2900 traj/s
# Write: latency = 26.4765 s; throughput = 553.3952 MB/s, 1.9262 traj/s

# Arrow_IPC:
# Disk size = 14651.9697 MB ; Num. traj = 51
# Read:  latency = 44.0609 s; throughput = 332.5391 MB/s, 1.1575 traj/s
# Write: latency = 25.1574 s; throughput = 582.4108 MB/s, 2.0272 traj/s

# Feather_FTH:
# Disk size = 14651.9697 MB; Num. traj = 51
# Read:  latency = 47.5646 s; throughput = 308.0433 MB/s, 1.0722 traj/s
# Write: latency = 35.4595 s; throughput = 413.2030 MB/s, 1.4383 traj/s

# Pandas_FTH:
# Disk size = 14651.9697 MB; Num. traj = 51
# Read:  latency = 26.6213 s; throughput = 550.3844 MB/s, 1.9158 traj/s
# Write: latency = 45.9015 s; throughput = 319.2046 MB/s, 1.1111 traj/s

"""###"""

# HDF5:
# Disk size = 44319.6689 MB  ; Num. traj = 51
# Read:  latency = 191.5964 s; throughput = 231.3179 MB/s, 0.2662 traj/s
# Write: latency = 129.6112 s; throughput = 341.9432 MB/s, 0.3935 traj/s


import matplotlib.pyplot as plt

# Josh's data for Polars
polars_read_latency = 43.7798
polars_read_speed = 334.6640
polars_read_throughput = 1.1649

polars_write_latency = 257.1229
polars_write_speed = 56.9826
polars_write_throughput = 0.1983

# above data for TensorFlow Record
tf_read_latency = 16.76
tf_read_speed = 484.53
tf_read_throughput = 6.21

tf_write_latency = 44.17
tf_write_speed = 183.84
tf_write_throughput = 2.35

labels = ['Polars Read', 'Polars Write', 'TF Record Read', 'TF Record Write']

latencies = [polars_read_latency, polars_write_latency, tf_read_latency, tf_write_latency]
speeds = [polars_read_speed, polars_write_speed, tf_read_speed, tf_write_speed]
throughputs = [polars_read_throughput, polars_write_throughput, tf_read_throughput, tf_write_throughput]

#  latency plots
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.bar(labels, latencies, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Latency (s)')
plt.title('Latency, Speed, and Throughput of Polars and TF Record Operations')

# speed plots
plt.subplot(3, 1, 2)
plt.bar(labels, speeds, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Speed (MB/s)')

#  throughput plotting
plt.subplot(3, 1, 3)
plt.bar(labels, throughputs, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Throughput (trajectories/second)')
plt.xlabel('Operation')

plt.tight_layout()
plt.show()
