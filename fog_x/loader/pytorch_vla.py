import torch
from torch.utils.data import Dataset, DataLoader
import fog_x
import glob
import os
import random
import asyncio
from typing import Text, List

class VLADataset(Dataset):
    def __init__(self, path: Text, cache_dir: Text = None, shuffle: bool = True):
        self.files = self._get_files(path)
        self.cache_dir = cache_dir
        self.loop = asyncio.get_event_loop()
        if shuffle:
            random.shuffle(self.files)

    def _get_files(self, path: Text) -> List[Text]:
        if "*" in path:
            return glob.glob(path)
        elif os.path.isdir(path):
            return glob.glob(os.path.join(path, "*.vla"))
        else:
            return [path]

    async def _read_vla_async(self, data_path: Text):
        traj = fog_x.Trajectory(data_path, cache_dir=self.cache_dir)
        data = await traj.load_async()
        return data

    def _read_vla(self, data_path: Text):
        return self.loop.run_until_complete(self._read_vla_async(data_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        file_path = self.files[index]
        data = self._read_vla(file_path)
        # Convert data to PyTorch tensors
        # You may need to adjust this based on the structure of your VLA data
        return data #{k: torch.tensor(v) for k, v in data.items()}

def vla_collate_fn(batch):
    # Implement custom collate function if needed
    # This depends on the structure of your VLA data
    return batch

def get_vla_dataloader(path: Text, batch_size: int = 1, cache_dir: Text = None, shuffle: bool = True, num_workers: int = 0):
    dataset = VLADataset(path, cache_dir, shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=vla_collate_fn)