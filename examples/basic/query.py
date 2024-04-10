import fog_x

# create a new dataset
dataset = fog_x.Dataset(
    name="test_rtx",
    path="/tmp/rtx",
)

# create a new episode / trajectory
episode = dataset.new_episode(
    metadata={
        "description": "grasp teddy bear from the shelf",
    }
)

# populate the episode with FeatureTypes
for image, pose in my_camera.read():
    # create a class builder pattern
    episode.add(feature="arm_view", value=f"image{i}.jpg")
    episode.add(feature="camera_pose", value=f"pose{i}")

# mark the current state as terminal state
# and save the episode
episode.close()

# Method 1:
# load the dataset metadata as a pandas dataframe
metadata = dataset.get_metadata_as_pandas_df()

# ...
# do what you want like a typical pandas dataframe
# Example: load with shuffled the episodes in the dataset
metadata = metadata.sample(frac=1)[:4]
episodes = dataset.load(metadata=metadata)
for episode in episodes:
    print(episode)

# Method 2:
# query the metadata with sql
metadata = dataset.get_metadata_as_sql()
metadata = metadata.query("description == 'grasp teddy bear from the shelf'")
episodes = dataset.load(metadata=metadata)

# export the dataset
dataset.export("/tmp/rtx_export", format="rtx")
