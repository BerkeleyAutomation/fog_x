
import fog_rtx

dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="/tmp",
)

# dataset.load_rtx_episodes(
#     name="berkeley_autolab_ur5",
#     split="train[:2]",
#     additional_metadata={"collector": "User 1"},
# )

# dataset.load_rtx_episodes(
#     name="berkeley_autolab_ur5",
#     split="train[3:5]",
#     additional_metadata={"collector": "User 2"},
# )


tf_ds = dataset.tensorflow_dataset_builder(
    metadata=dataset.get_metadata_as_pandas_df()
)

# get samples from the dataset
for data in tf_ds:
    print(data)