from fog_x.loader import VLALoader
import fog_x
import os


loader = VLALoader("/tmp/fog_x/vla/berkeley_autolab_ur5/*.vla")
for index, data_traj in enumerate(loader):

    print(data_traj.load())
    index += 1