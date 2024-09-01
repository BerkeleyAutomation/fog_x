import torch
from torch.utils.data import IterableDataset, DataLoader
from . import BaseLoader
import numpy as np 
import glob 
import h5py
import asyncio

# flatten the data such that all data starts with root level tree (observation and action)
def _flatten(data, parent_key='', sep='/'):
    items = {}
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items
        
def recursively_read_hdf5_group(group):
    if isinstance(group, h5py.Dataset):
        return np.array(group)
    elif isinstance(group, h5py.Group):
        return {key: recursively_read_hdf5_group(value) for key, value in group.items()}
    else:
        raise TypeError("Unsupported HDF5 group type")
    

class HDF5Loader(BaseLoader):
    def __init__(self, path, batch_size=1):
        super(HDF5Loader, self).__init__(path)
        self.index = 0
        self.files = glob.glob(self.path, recursive=True)
        self.batch_size = batch_size
    async def _read_hdf5_async(self, data_path):
        return await asyncio.to_thread(self._read_hdf5, data_path)

    async def get_batch(self):
        tasks = []
        for _ in range(self.batch_size):
            if self.index < len(self.files):
                file_path = self.files[self.index]
                self.index += 1
                tasks.append(self._read_hdf5_async(file_path))
            else:
                break
        return await asyncio.gather(*tasks)

    def __next__(self):
        if self.index >= len(self.files):
            self.index = 0
            raise StopIteration
        return asyncio.run(self.get_batch())

    def _read_hdf5(self, data_path):
        
        with h5py.File(data_path, "r") as f:
            data_unflattened = recursively_read_hdf5_group(f)

        data = {}
        data["observation"] = _flatten(data_unflattened["observation"])
        data["action"] = _flatten(data_unflattened["action"])

        return data

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.files)

class HDF5IterableDataset(IterableDataset):
    def __init__(self, path, batch_size=1):
        self.hdf5_loader = HDF5Loader(path, batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.hdf5_loader)
            return batch[0]  # Return a single item, not a batch
        except StopIteration:
            raise StopIteration

def hdf5_collate_fn(batch):
    # Convert data to PyTorch tensors
    return batch 

def get_hdf5_dataloader(
    path: str,
    batch_size: int = 1,
    num_workers: int = 0
):
    dataset = HDF5IterableDataset(path, batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=hdf5_collate_fn,
        num_workers=num_workers
    )

