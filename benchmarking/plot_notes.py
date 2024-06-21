# TF_Record: 
# Disk size = 8039.7179 MB  ; Num. traj = 51
# Read:  latency = 15.5696 s; throughput = 516.3728 MB/s, 3.2756 traj/s
# Write: latency = 36.6353 s; throughput = 219.4528 MB/s, 1.3921 traj/s

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

metrics = {
    "TF_Record": {
        "Disk size": 8039.7179,
        "Read latency": 15.5696,
        "Read throughput MB/s": 516.3728,
        "Read throughput traj/s": 3.2756,
        "Write latency": 36.6353,
        "Write throughput MB/s": 219.4528,
        "Write throughput traj/s": 1.3921,
    },
    "Fog-X": {
        "Disk size": 6707.3818,
        "Read latency": 42.1416,
        "Read throughput MB/s": 159.1630,
        "Read throughput traj/s": 1.2102,
        "Write latency": 258.8635,
        "Write throughput MB/s": 25.9109,
        "Write throughput traj/s": 0.1970,
    },
    "Polars": {
        "Disk size": 6707.3818,
        "Read latency": 22.9974,
        "Read throughput MB/s": 291.6581,
        "Read throughput traj/s": 2.2176,
        "Write latency": 241.4137,
        "Write throughput MB/s": 27.7838,
        "Write throughput traj/s": 0.2113,
    },
    "PyArrow": {
        "Disk size": 6707.3818,
        "Read latency": 27.9808,
        "Read throughput MB/s": 239.7136,
        "Read throughput traj/s": 1.8227,
        "Write latency": 80.9986,
        "Write throughput MB/s": 82.8086,
        "Write throughput traj/s": 0.6296,
    },
    "Pandas": {
        "Disk size": 6707.3818,
        "Read latency": 49.1056,
        "Read throughput MB/s": 136.5909,
        "Read throughput traj/s": 1.0386,
        "Write latency": 103.6820,
        "Write throughput MB/s": 64.6918,
        "Write throughput traj/s": 0.4919,
    },
    "Arrow_REG": {
        "Disk size": 14651.9697,
        "Read latency": 39.5354,
        "Read throughput MB/s": 370.6041,
        "Read throughput traj/s": 1.29,
        "Write latency": 26.4765,
        "Write throughput MB/s": 553.3952,
        "Write throughput traj/s": 1.9262,
    },
    "Arrow_IPC": {
        "Disk size": 14651.9697,
        "Read latency": 44.0609,
        "Read throughput MB/s": 332.5391,
        "Read throughput traj/s": 1.1575,
        "Write latency": 25.1574,
        "Write throughput MB/s": 582.4108,
        "Write throughput traj/s": 2.0272,
    },
    "Feather_FTH": {
        "Disk size": 14651.9697,
        "Read latency": 47.5646,
        "Read throughput MB/s": 308.0433,
        "Read throughput traj/s": 1.0722,
        "Write latency": 35.4595,
        "Write throughput MB/s": 413.203,
        "Write throughput traj/s": 1.4383,
    },
    "Pandas_FTH": {
        "Disk size": 14651.9697,
        "Read latency": 26.6213,
        "Read throughput MB/s": 550.3844,
        "Read throughput traj/s": 1.9158,
        "Write latency": 45.9015,
        "Write throughput MB/s": 319.2046,
        "Write throughput traj/s": 1.1111,
    },
    "HDF5": {
        "Disk size": 44319.6689,
        "Read latency": 191.5964,
        "Read throughput MB/s": 231.3179,
        "Read throughput traj/s": 0.2662,
        "Write latency": 129.6112,
        "Write throughput MB/s": 341.9432,
        "Write throughput traj/s": 0.3935,
    },
}

# Metrics to plot
metric_names = [
    "Disk size", 
    "Read latency", 
    "Read throughput MB/s", 
    "Read throughput traj/s", 
    "Write latency", 
    "Write throughput MB/s", 
    "Write throughput traj/s"
]

# Colors for each library
colors = {
    "TF_Record": "red",
    "Fog-X": "orange",
    "Polars": "yellow",
    "PyArrow": "green",
    "Pandas": "cyan",
    "Arrow_REG": "blue",
    "Arrow_IPC": "purple",
    "Feather_FTH": "pink",
    "Pandas_FTH": "brown",
    "HDF5": "gray",
}

# Create plots
for metric in metric_names:
    plt.figure(figsize=(10, 6))
    libraries = []
    values = []
    for library, stats in metrics.items():
        libraries.append(library)
        values.append(stats[metric])
    bars = plt.bar(libraries, values, color=[colors[lib] for lib in libraries])
    plt.xlabel('Library')
    plt.ylabel(metric)
    plt.title(f'{metric} by Library')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{metric.replace("/", "_")}.png')
    plt.close()