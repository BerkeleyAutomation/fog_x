

import fog_rtx

# create a new dataset
dataset = fog_rtx.dataset.Dataset(
    name="test_rtx",
    path="/tmp/rtx", 
    replace_existing=False, 
)  

# create a new episode / trajectory
episode = dataset.new_episode(
    description = "grasp teddy bear from the shelf"
)

# populate the episode with features
episode.add(feature = "arm_view", value = "image1.jpg")
episode.add(feature = "camera_pose", value = "image1.jpg")

# flush save the episode
episode.close()

# load the dataset
episodes = dataset.query("SELECT *")
for episode in episodes:
    print(episode)

# export the dataset
dataset.export("/tmp/rtx_export", format="rtx")