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

# Data for Fog-X
fogx_read_latency_mb = 42.1416
fogx_read_throughput_mb = 159.1630
fogx_read_throughput_traj = 1.2102
fogx_write_latency_mb = 258.8635
fogx_write_throughput_mb = 25.9109
fogx_write_throughput_traj = 0.1970

# Data for Polars
polars_read_latency_mb = 22.9974
polars_read_throughput_mb = 291.6581
polars_read_throughput_traj = 2.2176
polars_write_latency_mb = 241.4137
polars_write_throughput_mb = 27.7838
polars_write_throughput_traj = 0.2113

# Data for PyArrow
pyarrow_read_latency_mb = 27.9808
pyarrow_read_throughput_mb = 239.7136
pyarrow_read_throughput_traj = 1.8227
pyarrow_write_latency_mb = 80.9986
pyarrow_write_throughput_mb = 82.8086
pyarrow_write_throughput_traj = 0.6296

# Data for Pandas
pandas_read_latency_mb = 49.1056
pandas_read_throughput_mb = 136.5909
pandas_read_throughput_traj = 1.0386
pandas_write_latency_mb = 103.6820
pandas_write_throughput_mb = 64.6918
pandas_write_throughput_traj = 0.4919

# Plotting
labels = ['Fog-X', 'Polars', 'PyArrow', 'Pandas']
read_latencies = [fogx_read_latency_mb, polars_read_latency_mb, pyarrow_read_latency_mb, pandas_read_latency_mb]
read_throughputs_mb = [fogx_read_throughput_mb, polars_read_throughput_mb, pyarrow_read_throughput_mb, pandas_read_throughput_mb]
read_throughputs_traj = [fogx_read_throughput_traj, polars_read_throughput_traj, pyarrow_read_throughput_traj, pandas_read_throughput_traj]
write_latencies = [fogx_write_latency_mb, polars_write_latency_mb, pyarrow_write_latency_mb, pandas_write_latency_mb]
write_throughputs_mb = [fogx_write_throughput_mb, polars_write_throughput_mb, pyarrow_write_throughput_mb, pandas_write_throughput_mb]
write_throughputs_traj = [fogx_write_throughput_traj, polars_write_throughput_traj, pyarrow_write_throughput_traj, pandas_write_throughput_traj]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plotting read latency and throughput
axes[0, 0].bar(labels, read_latencies, color='blue')
axes[0, 0].set_title('Read Latency (s)')
axes[0, 1].bar(labels, read_throughputs_mb, color='green')
axes[0, 1].set_title('Read Throughput (MB/s)')
axes[0, 2].bar(labels, read_throughputs_traj, color='orange')
axes[0, 2].set_title('Read Throughput (traj/s)')

# Plotting write latency and throughput
axes[1, 0].bar(labels, write_latencies, color='blue')
axes[1, 0].set_title('Write Latency (s)')
axes[1, 1].bar(labels, write_throughputs_mb, color='green')
axes[1, 1].set_title('Write Throughput (MB/s)')
axes[1, 2].bar(labels, write_throughputs_traj, color='orange')
axes[1, 2].set_title('Write Throughput (traj/s)')

# Adding captions
captions = [
    f'Fog-X: Disk size = 6707.3818 MB, Num. traj = 51',
    f'Polars: Disk size = 6707.3818 MB, Num. traj = 51',
    f'PyArrow: Disk size = 6707.3818 MB, Num. traj = 51',
    f'Pandas: Disk size = 6707.3818 MB, Num. traj = 51'
]

for ax, caption in zip(axes.flatten(), captions):
    ax.annotate(caption, xy=(0.5, -0.3), xycoords='axes fraction',
                ha='center', va='center', fontsize=10)

# Adjust layout
plt.tight_layout()
plt.show()
