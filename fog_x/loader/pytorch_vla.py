import torch
from torch.utils.data import IterableDataset, DataLoader
from fog_x.loader.vla import VLALoader
from typing import Text, Optional

class VLAIterableDataset(IterableDataset):
    def __init__(self, path: Text, cache_dir: Optional[Text] = None, buffer_size: int = 1000):
        self.vla_loader = VLALoader(path, batch_size=1, cache_dir=cache_dir, buffer_size=buffer_size)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.vla_loader.get_batch()
        if batch is None:
            raise StopIteration
        return batch[0]  # Return a single item, not a batch

def vla_collate_fn(batch):
    # Convert data to PyTorch tensors
    # You may need to adjust this based on the structure of your VLA data
    return batch #{k: torch.tensor(v) for k, v in batch[0].items()}

def get_vla_dataloader(
    path: Text,
    batch_size: int = 1,
    cache_dir: Optional[Text] = None,
    buffer_size: int = 1000,
    num_workers: int = 0
):
    dataset = VLAIterableDataset(path, cache_dir, buffer_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=vla_collate_fn,
        num_workers=num_workers
    )