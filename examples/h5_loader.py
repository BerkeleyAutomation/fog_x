from fog_x.loader.hdf5 import HDF5Loader
import fog_x

import os
os.system("rm -rf /tmp/fog_x/*")

loader = HDF5Loader("/home/kych/datasets/2024-07-03-red-on-cyan/**/trajectory_im128.h5")

index = 0

for data_traj in loader:

    fog_x.Trajectory.from_dict_of_lists(
        data_traj, path=f"/tmp/fog_x/output_{index}.vla"
    )
    index += 1
    

# read the data back
for i in range(index):
    print(fog_x.Trajectory(f"/tmp/fog_x/output_{i}.vla")["action"].keys())