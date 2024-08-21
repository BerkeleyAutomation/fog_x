import fog_x
import numpy as np
import time 

path = "/tmp/output.vla"
# remove the existing file
import os
os.system(f"rm -rf {path}")
os.system(f"rm -rf /tmp/*.cache")

# 🦊 Data collection: 
# create a new trajectory
traj = fog_x.Trajectory(
    path = path
)

traj.init_feature_streams(
    feature_spec = {
        "arm_view": fog_x.FeatureType(dtype="uint8", shape=(640, 480, 3)),
        "gripper_pose": fog_x.FeatureType(dtype="float32", shape=(4, 4)),
        "view": fog_x.FeatureType(dtype="uint8", shape=(640, 480, 3)),
        "wrist_view": fog_x.FeatureType(dtype="uint8", shape=(640, 480, 3)),
        "joint_angles": fog_x.FeatureType(dtype="float32", shape=(7,)),
        "joint_velocities": fog_x.FeatureType(dtype="float32", shape=(7,)),
        "joint_torques": fog_x.FeatureType(dtype="float32", shape=(7,)),
        "ee_pose": fog_x.FeatureType(dtype="float32", shape=(4, 4)),
        "ee_velocity": fog_x.FeatureType(dtype="float32", shape=(6,)),
        "ee_force": fog_x.FeatureType(dtype="float32", shape=(6,)),
    }
)

# collect step data for the episode
for i in range(100):
    time.sleep(0.001)
    traj.add(feature = "arm_view", data = np.ones((640, 480, 3), dtype=np.uint8))
    traj.add(feature = "gripper_pose", data = np.ones((4, 4), dtype=np.float32))
    traj.add(feature = "view", data = np.ones((640, 480, 3), dtype=np.uint8))
    traj.add(feature = "wrist_view", data = np.ones((640, 480, 3), dtype=np.uint8))
    traj.add(feature = "joint_angles", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "joint_velocities", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "joint_torques", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "ee_pose", data = np.ones((4, 4), dtype=np.float32))
    traj.add(feature = "ee_velocity", data = np.ones((6,), dtype=np.float32))
    traj.add(feature = "ee_force", data = np.ones((6,), dtype=np.float32))

# Automatically time-aligns and saves the trajectory
traj.close()


traj = fog_x.Trajectory(
    path = path
)