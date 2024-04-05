import fog_rtx

dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="/tmp",
)

dataset.load(
    dataset_path="berkeley_autolab_ur5",
    format="rtx",
    split="train[:5]",
)
