# Fog-X:
# Data size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 42.1416 s ; throughput = 159.1630 MB/s, 1.2102 traj/s
# Write: latency = 258.8635 s; throughput = 25.9109 MB/s,  0.1970 traj/s

# Polars: 
# Data size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 22.9974 s ; throughput = 291.6581 MB/s, 2.2176 traj/s
# Write: latency = 241.4137 s; throughput = 27.7838 MB/s,  0.2113 traj/s

# PyArrow:
# Data size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 27.9808 s ; throughput = 239.7136 MB/s, 1.8227 traj/s
# Write: latency = 80.9986 s ; throughput = 82.8086 MB/s,  0.6296 traj/s

# Pandas: 
# Data size = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 49.1056 s ; throughput = 136.5909 MB/s, 1.0386 traj/s
# Write: latency = 103.6820 s; throughput = 64.6918 MB/s,  0.4919 traj/s

# plotting + code
# temporary plots:

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
