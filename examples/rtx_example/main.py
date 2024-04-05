
import fog_rtx
dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="/tmp",
)

dataset.load(
    dataset_path="fractal20220817_data",
    format="rtx",
    split="train[:5]",
)