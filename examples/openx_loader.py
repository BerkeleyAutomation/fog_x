from fog_x.loader import RLDSLoader
import fog_x

import os

data_dir = "/home/kych/datasets/rtx"
dataset_name = "berkeley_autolab_ur5"
dataset_name = "fractal20220817_data"

# loader = RLDSLoader(
#     path="/home/kych/datasets/rtx/berkeley_autolab_ur5/0.1.0", split="train[:100]"
# )

loader = RLDSLoader(
    path=f"{data_dir}/{dataset_name}/0.1.0", split="train"
)

from concurrent.futures import ProcessPoolExecutor
import os

def process_data(data_traj, dataset_name, index):
    fog_x.Trajectory.from_list_of_dicts(
        data_traj, path=f"/home/kych/datasets/fog_x/vla/{dataset_name}/output_{index}.vla"
    )

# Use ThreadPoolExecutor to parallelize the processing
with ProcessPoolExecutor() as executor:
    futures = []
    for index, data_traj in enumerate(loader):
        # Submit the task to the executor
        futures.append(executor.submit(process_data, data_traj, dataset_name, index))

    # Optionally, wait for all futures to complete
    for future in futures:
        future.result()  # This will raise an exception if the task raised one

print("All tasks completed.")