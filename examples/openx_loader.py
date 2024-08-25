from fog_x.loader import RLDSLoader
import fog_x

import os

os.system("rm -rf /tmp/fog_x/*")

loader = RLDSLoader(
    path="/home/kych/datasets/rtx/berkeley_autolab_ur5/0.1.0", split="train[:10]"
)

index = 0

for data_traj in loader:

    fog_x.Trajectory.from_list_of_dicts(
        data_traj, path=f"/tmp/fog_x/output_{index}.vla"
    )
    index += 1
