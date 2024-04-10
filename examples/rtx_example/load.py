import fog_x

dataset = fog_x.dataset.Dataset(
    name="demo_ds",
    path="~/test_dataset",
)

dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    split="train[:1]",
)

dataset.export(format="rtx")
