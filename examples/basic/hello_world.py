import fog_x
import numpy as np
# 🦊 Data collection: 
# create a new trajectory
traj = fog_x.Trajectory(
    path = "/tmp/output.mkv"
)

# collect step data for the episode
for i in range(100):
    traj.add(feature = "arm_view", data = np.ones((640, 480, 3), dtype=np.uint8))
# Automatically time-aligns and saves the trajectory
traj.close()
