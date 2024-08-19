import fog_x

# 🦊 Data collection: 
# create a new trajectory
traj = fog_x.Trajectory(
    path = "/tmp/a.mkv"
)
# collect step data for the episode
traj.add(feature = "arm_view", value = "image1.jpg")
# Automatically time-aligns and saves the trajectory
traj.close()
