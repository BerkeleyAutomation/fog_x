from fog_x.loader.base import BaseLoader
import fog_x
import glob
import logging
import asyncio
import os
from typing import Text

logger = logging.getLogger(__name__)

class VLALoader(BaseLoader):
    def __init__(self, path: Text, cache_dir=None):
        super(VLALoader, self).__init__(path)
        self.index = 0
        self.files = self._get_files(path)
        self.cache_dir = cache_dir
        self.loop = asyncio.get_event_loop()

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
        await traj.load_async()
        return traj

    def _read_vla(self, data_path):
        return self.loop.run_until_complete(self._read_vla_async(data_path))

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.files):
            file_path = self.files[self.index]
            self.index += 1
            return self._read_vla(file_path)
        raise StopIteration

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self._read_vla(self.files[index])
    
    def peak(self):
        return self._read_vla(self.files[self.index])