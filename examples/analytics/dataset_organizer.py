import fog_x

dataset = fog_x.dataset.Dataset(
    name="demo_ds",
    path="~/test_dataset",
)

dataset._prepare_rtx_metadata(
    name="berkeley_autolab_ur5",
)


