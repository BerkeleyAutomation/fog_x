from fog_x.loader.base import BaseLoader
import fog_x
import glob
import logging

logger = logging.getLogger(__name__)
import os
from typing import Text


class VLALoader(BaseLoader):
    def __init__(self, path: Text, cache_dir=None):
        """initialize VLALoader from paths

        Args:
            path (_type_): path to the vla files
                    can be a directory, or a glob pattern
            split (_type_, optional): split of training and testing. Defaults to None.
        """
        super(VLALoader, self).__init__(path)
        self.index = 0

        if "*" in path:
            self.files = glob.glob(path)
        elif os.path.isdir(path):
            self.files = glob.glob(os.path.join(path, "*.vla"))
        else:
            self.files = [path]
        
        self.cache_dir = cache_dir

    def _read_vla(self, data_path):
        logger.info(f"Reading {data_path}")
        if self.cache_dir:
            traj = fog_x.Trajectory(data_path, cache_dir=self.cache_dir)
        else:
            traj = fog_x.Trajectory(data_path)
        return traj

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
