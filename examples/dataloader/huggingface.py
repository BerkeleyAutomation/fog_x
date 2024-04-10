import fog_rtx

dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="~/test_dataset",
)

dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    split="train[:1]",
)

huggingface_ds = dataset.get_as_huggingface_dataset()

print(f"Hugging face dataset: {huggingface_ds}")