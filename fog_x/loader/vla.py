from fog_x.loader.base import BaseLoader
import fog_x
import glob
import logging
import asyncio
import os
from typing import Text
import random
logger = logging.getLogger(__name__)

class VLALoader(BaseLoader):
    def __init__(self, path: Text, batch_size=1, cache_dir=None, shuffle=True):
        super(VLALoader, self).__init__(path)
        self.index = 0
        self.files = self._get_files(path)
        self.cache_dir = cache_dir
        self.loop = asyncio.get_event_loop()
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.files)

    def _get_files(self, path):
        if "*" in path:
            return glob.glob(path)
        elif os.path.isdir(path):
            return glob.glob(os.path.join(path, "*.vla"))
        else:
            return [path]

    async def _read_vla_async(self, data_path):
        logger.debug(f"Reading {data_path}")
        traj = fog_x.Trajectory(data_path, cache_dir=self.cache_dir)
        data = await traj.load_async()
        return data

    def _read_vla(self, data_path):
        return self.loop.run_until_complete(self._read_vla_async(data_path))

    async def get_batch_async(self):
        tasks = []
        for _ in range(self.batch_size):
            if self.index < len(self.files):
                file_path = self.files[self.index]
                self.index += 1
                tasks.append(self._read_vla_async(file_path))
            else:
                break
        batch_tasks = await asyncio.gather(*tasks)
        return batch_tasks
    
    
    def get_batch(self):
        batch = self.loop.run_until_complete(self.get_batch_async())
        return batch
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.files):
            self.index = 0
            if self.shuffle:
                random.shuffle(self.files)
            raise StopIteration
        
        return self.get_batch()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self._read_vla(self.files[index])
    
    def peak(self):
        return self._read_vla(self.files[self.index])