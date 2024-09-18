from fog_x.loader.base import BaseLoader
import fog_x
import glob
import logging
import asyncio
import os
from typing import Text, List, Any
import random
from collections import deque
import multiprocessing as mp
import time
from multiprocessing import Manager

logger = logging.getLogger(__name__)

class VLALoader:
    def __init__(self, path: Text, batch_size=1, cache_dir="/tmp/fog_x/cache/", buffer_size=50, num_workers=-1, return_type = "numpy", split="all"):
        self.files = self._get_files(path, split)
        self.split = split
        
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.return_type = return_type
        # TODO: adjust buffer size
        # if "autolab" in path:
        #     self.buffer_size = 4
        self.buffer_size = buffer_size
        self.buffer = mp.Queue(maxsize=buffer_size)
        if num_workers == -1:
            num_workers = 2
        self.num_workers = num_workers
        self.processes = []
        random.shuffle(self.files)
        self._start_workers()
        
    def _get_files(self, path, split):
        ret = []
        if "*" in path:
            ret = glob.glob(path)
        elif os.path.isdir(path):
            ret = glob.glob(os.path.join(path, "*.vla"))
        else:
            ret = [path]
        if split == "train":
            ret = ret[:int(len(ret)*0.9)]
        elif split == "val":
            ret = ret[int(len(ret)*0.9):]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Invalid split: {split}")
        return ret
    
    def _read_vla(self, data_path, return_type = None):
        if return_type is None:
            return_type = self.return_type
        traj = fog_x.Trajectory(data_path, cache_dir=self.cache_dir)
        ret = traj.load(return_type = return_type)
        return ret

    def _worker(self):
        max_retries = 3
        while True:
            if not self.files:
                logger.info("Worker finished")
                break
            
            for attempt in range(max_retries):
                try:
                    file_path = random.choice(self.files)
                    data = self._read_vla(file_path)
                    self.buffer.put(data)
                    break  # Exit the retry loop if successful
                except Exception as e:
                    logger.error(f"Error reading {file_path} on attempt {attempt + 1}: {e}")
                    if attempt + 1 == max_retries:
                        logger.error(f"Failed to read {file_path} after {max_retries} attempts")

    def _start_workers(self):
        for _ in range(self.num_workers):
            p = mp.Process(target=self._worker)
            p.start()
            logger.debug(f"Started worker {p.pid}")
            self.processes.append(p)

    def get_batch(self) -> List[Any]:
        batch = []
        timeout = 5  # Adjust this value based on your needs
        start_time = time.time()

        while len(batch) < self.batch_size:
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout reached while getting batch. Batch size: {len(batch)}")
                break

            try:
                item = self.buffer.get(timeout=1)
                batch.append(item)
            except mp.queues.Empty:
                if all(not p.is_alive() for p in self.processes) and self.buffer.empty():
                    if len(batch) == 0:
                        return None  # No more data available
                    else:
                        break  # Return partial batch

        return batch

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.get_batch()
        if batch is None:
            random.shuffle(self.files)
            self._start_workers()
            raise StopIteration
        return batch

    def __len__(self):
        return len(self.files)

    def peek(self):
        file = random.choice(self.files)
        return self._read_vla(file, return_type = "numpy")

    def __del__(self):
        for p in self.processes:
            p.terminate()
            p.join()
            
            
class NonShuffleVLALoader:
    def __init__(self, path: Text, batch_size=1, cache_dir="/tmp/fog_x/cache/", num_workers=1, return_type = "numpy"):
        self.files = self._get_files(path)
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.return_type = return_type
        self.index = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.files):
            raise StopIteration
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(self.index)
                file_path = self.files[self.index]
                self.index += 1
                return self._read_vla(file_path, return_type = self.return_type)
            except Exception as e:
                logger.error(f"Error reading {file_path} on attempt {attempt + 1}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(f"Failed to read {file_path} after {max_retries} attempts")
                    return None

    def _get_files(self, path):
        ret = []
        if "*" in path:
            ret = glob.glob(path)
        elif os.path.isdir(path):
            ret = glob.glob(os.path.join(path, "*.vla"))
        else:
            ret = [path]
        # for file in ret:
        #     try:
        #         self._read_vla(file, return_type = self.return_type)
        #     except Exception as e:
        #         logger.error(f"Error reading {file}: {e}, ")
        #         ret.remove(file)
        return ret
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return self.files[index]
    
    def __del__(self):
        pass
    
    def peek(self):
        file = self.files[self.index]
        return self._read_vla(file, return_type = "numpy")

    def _read_vla(self, data_path, return_type = None):
        if return_type is None:
            return_type = self.return_type
        traj = fog_x.Trajectory(data_path, cache_dir=self.cache_dir)
        ret = traj.load(return_type = return_type)
        return ret
    
    def get_batch(self):
        return [self.__next__() for _ in range(self.batch_size)]

import torch
from torch.utils.data import IterableDataset, DataLoader
from fog_x.loader.vla import VLALoader
from typing import Text, Optional

class VLAIterableDataset(IterableDataset):
    def __init__(self, path: Text, cache_dir: Optional[Text] = None, buffer_size: int = 1000):
        # Note: batch size = 1 is to bypass the dataloader without pytorch dataloader 
        # in this case, we use pytorch dataloader for batching
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