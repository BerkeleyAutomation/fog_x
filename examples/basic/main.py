import fog_rtx

# create a new dataset
dataset = fog_rtx.dataset.Dataset(
    name="test_rtx",
    path="/tmp/rtx",
    replace_existing=False,
    db_connector=fog_rtx.database.DatabaseConnector("/tmp/rtx.db"),
)

for i in range(1,10):
    # create a new episode / trajectory
    episode = dataset.new_episode(
        metadata={
            "collector_name": f"User #{i}",
            "description": f"description #{i}",
        }
    )
    # populate the episode with FeatureTypes
    for j in range(1, 4):
        episode.add(feature="feature_1", value=f"episode{i}_step{j}_feature_1")
        episode.add(feature="feature_2", value=f"episode{i}_pose{j}_feature_2")
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
