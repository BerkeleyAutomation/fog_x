

import fog_rtx

# create a new dataset
dataset = fog_rtx.dataset.Dataset(
    name="test_rtx",
    path="/tmp/rtx", 
    replace_existing=False, 
    features = {
        "arm_view":  fog_rtx.feature.FeatureType(dtype="image"),
        "camera_pose": fog_rtx.feature.FeatureType(dtype="float64", dimension=[4, 4]),
    }, 
    db_connector = fog_rtx.database.SQLite("/tmp/rtx.db"), 
)  

# create a new episode / trajectory
episode = dataset.new_episode(
    description = "grasp teddy bear from the shelf"
)

# populate the episode with FeatureTypes
episode.add(feature = "arm_view", value = "image1.jpg")
episode.add(feature = "camera_pose", value = "image1.jpg")

# mark the current state as terminal state 
# and save the episode
episode.close()

# load the dataset
episodes = dataset.query("SELECT *")
for episode in episodes:
    print(episode)

# export the dataset
dataset.export("/tmp/rtx_export", format="rtx")