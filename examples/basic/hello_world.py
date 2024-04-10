import fog_x

# create a new dataset
dataset = fog_x.Dataset(
    name="test_rtx",
    path="/tmp/rtx",
)

# create a new episode / trajectory
trajectory = dataset.new_episode()
trajectory.add(feature="hello", value=1.0)
trajectory.add(feature="world", value=2.0)
trajectory.close()

# iterate through the dataset
for episode in dataset.read_by(
    pandas_metadata=dataset.get_metadata_as_pandas_df()
):
    print(episode)

# export the dataset as standard rt-x format
dataset.export(
    "/tmp/rtx_export", format="rtx", obs_keys=["hello"], act_keys=["world"]
)
