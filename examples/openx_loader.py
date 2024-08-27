from fog_x.loader import RLDSLoader
import fog_x

import os

data_dir = "/home/kych/datasets/rtx"
dataset_name = "berkeley_autolab_ur5"
dataset_name = "berkeley_cable_routing"

# loader = RLDSLoader(
#     path="/home/kych/datasets/rtx/berkeley_autolab_ur5/0.1.0", split="train[:100]"
# )

loader = RLDSLoader(
    path=f"{data_dir}/{dataset_name}/0.1.0", split="train"
)


index = 0

for data_traj in loader:

    fog_x.Trajectory.from_list_of_dicts(
        data_traj, path=f"/home/kych/datasets/fog_x/vla/{dataset_name}/output_{index}.vla"
    )
    index += 1
