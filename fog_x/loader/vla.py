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

logger = logging.getLogger(__name__)

class VLALoader:
    def __init__(self, path: Text, batch_size=1, cache_dir=None, buffer_size=100, num_workers=4):
        self.files = self._get_files(path)
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = mp.Queue(maxsize=buffer_size)
        self.num_workers = num_workers
        self.processes = []
        random.shuffle(self.files)
        self._start_workers()

    def _get_files(self, path):
        if "*" in path:
            return glob.glob(path)
        elif os.path.isdir(path):
            return glob.glob(os.path.join(path, "*.vla"))
        else:
            return [path]

    def _read_vla(self, data_path):
        traj = fog_x.Trajectory(data_path, cache_dir=self.cache_dir)
        return traj.load()

    def _worker(self):
        while True:
            if not self.files:
                logger.info("Worker finished")
                break
            file_path = random.choice(self.files)
            data = self._read_vla(file_path)
            self.buffer.put(data)

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
        if self.buffer.empty():
            return None
        return self.buffer.get()

    def __del__(self):
        for p in self.processes:
            p.terminate()
            p.join()