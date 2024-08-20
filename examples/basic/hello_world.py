import fog_x
import numpy as np
import time 

path = "/tmp/output.mkv"
# remove the existing file
import os
os.system(f"rm -rf {path}")


# 🦊 Data collection: 
# create a new trajectory
traj = fog_x.Trajectory(
    path = path
)


# collect step data for the episode
for i in range(100):
    time.sleep(0.001)
    # traj.add(feature = "arm_view", data = np.ones((640, 480, 3), dtype=np.uint8))
    # traj.add(feature = "view", data = np.ones((640, 480, 3), dtype=np.uint8))
    # traj.add(feature = "wrist_view", data = np.ones((640, 480, 3), dtype=np.uint8))
    traj.add(feature = "gripper_pose", data = np.ones((4, 4), dtype=np.float32))
    traj.add(feature = "joint_angles", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "joint_velocities", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "joint_torques", data = np.ones((7,), dtype=np.float32))
    # traj.add(feature = "ee_pose", data = np.ones((4, 4), dtype=np.float32))
    # traj.add(feature = "ee_velocity", data = np.ones((6,), dtype=np.float32))
    # traj.add(feature = "ee_force", data = np.ones((6,), dtype=np.float32))
    # traj.add(feature = "contact", data = np.ones((1,), dtype=np.bool))

# Automatically time-aligns and saves the trajectory
traj.close()


traj = fog_x.Trajectory(
    path = path
)