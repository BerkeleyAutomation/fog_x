from fog_x.loader import RLDSLoader
import fog_x

import os

data_dir = "/home/kych//datasets/rtx"
dataset_name = "berkeley_cable_routing"
dataset_name = "berkeley_autolab_ur5"
dataset_name = "fractal20220817_data"
destination_dir = "/mnt/data/datasets/fog_x/ffv1"
# destination_dir = "/home/kych//datasets/fog_x/vla"
version = "0.1.0"

# loader = RLDSLoader(
#     path="/home/kych/datasets/rtx/berkeley_autolab_ur5/0.1.0", split="train[:100]"
# )

loader = RLDSLoader(
    path=f"{data_dir}/{dataset_name}/{version}", split="train"
)

from concurrent.futures import ProcessPoolExecutor
import os

def process_data(data_traj, dataset_name, index):
    try:
        fog_x.Trajectory.from_list_of_dicts(
            data_traj, path=f"{destination_dir}/{dataset_name}/output_{index}.vla"
        )
    except Exception as e:
        print(f"Failed to process data {index}: {e}")

# Use ThreadPoolExecutor to parallelize the processing
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    try:
        for index, data_traj in enumerate(loader):
            # Submit the task to the executor
            futures.append(executor.submit(process_data, data_traj, dataset_name, index))
    except Exception as e:
        print(f"Failed to process data {index}: {e}")

    # Optionally, wait for all futures to complete
    for future in futures:
        future.result()  # This will raise an exception if the task raised one

print("All tasks completed.")