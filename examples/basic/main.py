import fog_rtx

# create a new dataset
dataset = fog_rtx.dataset.Dataset(
    name="test_rtx",
    path="/tmp/rtx",
    replace_existing=False,
    features={
        "arm_view": fog_rtx.feature.FeatureType(dtype="image"),
        "camera_pose": fog_rtx.feature.FeatureType(
            dtype="float64", dimension=[4, 4]
        ),
    },
    db_connector=fog_rtx.database.DatabaseConnector("/tmp/rtx.db"),
)

# create a new episode / trajectory
episode = dataset.new_episode(description="grasp teddy bear from the shelf")

# populate the episode with FeatureTypes
for i in range(10):
    episode.add(feature="arm_view", value=f"image{i}.jpg")
    episode.add(feature="camera_pose", value=f"pose{i}")

# mark the current state as terminal state
# and save the episode
episode.close()

# load the dataset
metadata = dataset.get_metadata_as_pandas_df()
# ... 
# do what you want like a typical pandas dataframe
# Example: load with shuffled the episodes in the dataset
metadata = metadata.sample(frac = 1)
episodes = dataset.load_from(metadata)
for episode in episodes:
    print(episode)

# export the dataset
dataset.export("/tmp/rtx_export", format="rtx")
