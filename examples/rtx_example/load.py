import fog_rtx

dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="~/test_dataset",
)

dataset.load_rtx_episodes(
    name="berkeley_autolab_ur5",
    split="train[:1]",
)

dataset.export(format="rtx")
