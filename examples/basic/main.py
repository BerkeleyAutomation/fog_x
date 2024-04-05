import fog_rtx

# create a new dataset
dataset = fog_rtx.dataset.Dataset(
    name="test_rtx",
    path="/tmp/rtx",
    replace_existing=False,
    db_connector=fog_rtx.database.DatabaseConnector("/tmp/rtx.db"),
)

for i in range(10):
    # create a new episode / trajectory
    episode = dataset.new_episode(
        metadata={
            "collector_name": "John Doe",
            "description": f"description #{i}",
        }
    )

    # populate the episode with FeatureTypes
    for i in range(10):
        episode.add(feature="arm_view", value=f"image{i}.jpg")
        episode.add(feature="camera_pose", value=f"pose{i}")
    episode.close()

# mark the current state as terminal state
# and save the episode
episode.close()

# load the dataset
metadata = dataset.get_metadata_as_pandas_df()
# ...
# do what you want like a typical pandas dataframe
# Example: load with shuffled the episodes in the dataset
metadata = metadata.sample(frac=1)
print(metadata)
episodes = dataset.read_by(metadata)
for episode in episodes:
    print(episode)

# export the dataset
dataset.export("/tmp/rtx_export", format="rtx")
