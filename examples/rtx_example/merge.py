import fog_rtx

dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="~/test_dataset",
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
episode_info = dataset.get_episode_info()
print(episode_info)
# only get the episodes with custom_tag == "Partition_1"
metadata = episode_info.filter(episode_info["custom_tag"] == "Partition_1")
episodes = dataset.read_by(metadata)

# read the episodes
for episode in episodes:
    print(episode)
