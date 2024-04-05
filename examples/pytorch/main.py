import fog_rtx
import torch 
dataset = fog_rtx.dataset.Dataset(
    name="demo_ds",
    path="/tmp",
)

# dataset.load_rtx_episodes(
#     name="berkeley_autolab_ur5",
#     split="train[:1]",
# )


pytorch_ds = dataset.pytorch_dataset_builder()

# get samples from the dataset
for data in torch.utils.data.DataLoader(pytorch_ds, batch_size=2, collate_fn=lambda x: x, sampler = torch.utils.data.RandomSampler(pytorch_ds)):
    print(data)