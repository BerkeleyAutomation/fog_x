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

class HDF5EpisodeLoader(BaseLoader):
    def __init__(self, path, batch_size=1, buffer_size=50, num_workers=4):
        super(HDF5EpisodeLoader, self).__init__(path)
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

    def get_batch_by_episode(self):
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
        batch = self.get_batch_by_episode()
        if batch is None:
            random.shuffle(self.files)
            self._start_workers()
            raise StopIteration
        return batch

    def _read_hdf5(self, data_path):
        with h5py.File(data_path, "r") as f:
            data_unflattened = recursively_read_hdf5_group(f)
        print(data_unflattened.keys())
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


class HDF5FrameLoader(BaseLoader):
    def __init__(self, path, batch_size=1, buffer_size=50, num_workers=4, slice_size=1):
        super(HDF5FrameLoader, self).__init__(path)
        self.files = glob.glob(self.path, recursive=True)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = mp.Queue(maxsize=buffer_size)
        self.num_workers = num_workers
        self.processes = []
        self.slice_size = slice_size
        random.shuffle(self.files)
        self._start_workers()

    def _start_workers(self):
        for _ in range(self.num_workers):
            p = mp.Process(target=self._worker)
            p.start()
            logging.debug(f"Started worker {p.pid}")
            self.processes.append(p)
            
    def _worker(self):
        while True:
            if not self.files:
                logging.info("Worker finished")
                break
            file_path = random.choice(self.files)
            data = self._read_hdf5_slice(file_path)
            self.buffer.put(data)

    def _read_hdf5_slice(self, data_path):
        with h5py.File(data_path, "r") as f:
            total_frames = self.get_number_of_frames_in_episode(data_path)
            if self.slice_size > total_frames:
                start_idx = 0
                end_idx = total_frames
            else:
                start_idx = random.randint(0, total_frames - self.slice_size)
                end_idx = start_idx + self.slice_size
            
            slice_data = {}
            for key in f['observation'].keys():
                slice_data[f'observation/{key}'] = f[f'observation/{key}'][start_idx:end_idx]
            for key in f['action'].keys():
                slice_data[f'action/{key}'] = f[f'action/{key}'][start_idx:end_idx]

        return slice_data

    def get_number_of_frames_in_episode(self, data_path):
        with h5py.File(data_path, "r") as f:
            # get the first key that has the image data
            image_key = next((key for key in f['observation'].keys() if 'image' in key), None)
            if image_key is None:
                raise ValueError("No image data found in the dataset")
            return f['observation'][image_key].shape[0]

    def get_batch_by_slice(self):
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
        batch = self.get_batch_by_slice()
        if batch is None:
            random.shuffle(self.files)
            self._start_workers()
            raise StopIteration
        return batch

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


class HDF5IterableEpisodeDataset(IterableDataset):
    def __init__(self, path, batch_size=1):
        # Note: batch size = 1 is to bypass the dataloader without pytorch dataloader
        self.hdf5_loader = HDF5EpisodeLoader(path, 1)

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

class HDF5IterableFrameDataset(IterableDataset):
    def __init__(self, path, batch_size=1, slice_size=1):
        # Note: batch size = 1 is to bypass the dataloader without pytorch dataloader
        self.hdf5_loader = HDF5FrameLoader(path, batch_size=1, slice_size=slice_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.hdf5_loader)
            return batch[0]  # Return a single item, not a batch
        except StopIteration:
            raise StopIteration

def get_hdf5_dataloader(path: str, batch_size: int = 1, num_workers: int = 0, unit: str = "trajectory", slice_size: int = 1):
    if unit == "trajectory":
        dataset = HDF5IterableEpisodeDataset(path, batch_size)
    elif unit == "frame":
        dataset = HDF5IterableFrameDataset(path, batch_size, slice_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=hdf5_collate_fn,
        num_workers=num_workers,
    )
