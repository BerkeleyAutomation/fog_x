import fog_rtx

dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="/tmp",
)

dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    split="train[:2]",
    additional_metadata={"collector": "User 1", "custom_tag": "Partition_1"},
)

dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    split="train[3:5]",
    additional_metadata={"collector": "User 2", "custom_tag": "Partition_2"},
)
# dataset.num_episodes == 4

# query the dataset
metadata = dataset.get_metadata_as_pandas_df()
print(metadata)
# only get the episodes with custom_tag == "Partition_1"
metadata = metadata.filter(metadata["custom_tag"] == "Partition_1")
episodes = dataset.read_by(metadata)

# read the episodes
for episode in episodes:
    print(episode)
