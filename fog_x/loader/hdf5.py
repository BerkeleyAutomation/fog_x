import torch
from torch.utils.data import IterableDataset, DataLoader
from . import BaseLoader
import numpy as np
import glob
import h5py
import asyncio
import random
import multiprocessing as mp
import time
import logging
from fog_x.utils import _flatten, recursively_read_hdf5_group

class HDF5Loader(BaseLoader):
    def __init__(self, path, batch_size=1, buffer_size=100, num_workers=4):
        super(HDF5Loader, self).__init__(path)
        self.files = glob.glob(self.path, recursive=True)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = mp.Queue(maxsize=buffer_size)
        self.num_workers = num_workers
        self.processes = []
        random.shuffle(self.files)
        self._start_workers()

    def _worker(self):
        while True:
            if not self.files:
                logging.info("Worker finished")
                break
            file_path = random.choice(self.files)
            data = self._read_hdf5(file_path)
            self.buffer.put(data)

    def _start_workers(self):
        for _ in range(self.num_workers):
            p = mp.Process(target=self._worker)
            p.start()
            logging.debug(f"Started worker {p.pid}")
            self.processes.append(p)

    def get_batch(self):
        batch = []
        timeout = 5
        start_time = time.time()

        while len(batch) < self.batch_size:
            if time.time() - start_time > timeout:
                logging.warning(
                    f"Timeout reached while getting batch. Batch size: {len(batch)}"
                )
                break

            try:
                item = self.buffer.get(timeout=1)
                batch.append(item)
            except mp.queues.Empty:
                if (
                    all(not p.is_alive() for p in self.processes)
                    and self.buffer.empty()
                ):
                    if len(batch) == 0:
                        return None
                    else:
                        break
        return batch

    def __next__(self):
        batch = self.get_batch()
        if batch is None:
            random.shuffle(self.files)
            self._start_workers()
            raise StopIteration
        return batch

    def _read_hdf5(self, data_path):
        with h5py.File(data_path, "r") as f:
            data_unflattened = recursively_read_hdf5_group(f)

        data = {}
        data["observation"] = _flatten(data_unflattened["observation"])
        data["action"] = _flatten(data_unflattened["action"])

        return data_unflattened

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

    def peek(self):
        if self.buffer.empty():
            return None
        return self.buffer.get()

    def __del__(self):
        for p in self.processes:
            p.terminate()
            p.join()


class HDF5IterableDataset(IterableDataset):
    def __init__(self, path, batch_size=1):
        # Note: batch size = 1 is to bypass the dataloader without pytorch dataloader
        self.hdf5_loader = HDF5Loader(path, 1)

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


def get_hdf5_dataloader(path: str, batch_size: int = 1, num_workers: int = 0):
    dataset = HDF5IterableDataset(path, batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=hdf5_collate_fn,
        num_workers=num_workers,
    )
