"""VLA (Vision-Language-Action) data loaders for robodm."""

import glob
import logging
import multiprocessing as mp
import os
import random
import time
from typing import Any, List, Optional, Text

import torch
from torch.utils.data import DataLoader, IterableDataset

from robodm import Trajectory

from . import BaseLoader

logger = logging.getLogger(__name__)


class VLALoader(BaseLoader):
    """Shuffling VLA loader with multiprocessing support for random data loading."""

    def __init__(
        self,
        path: Text,
        batch_size: int = 1,
        cache_dir: Optional[Text] = None,
        buffer_size: int = 50,
        num_workers: int = -1,
        return_type: str = "numpy",
        split: str = "train"
    ):
        """Initialize the shuffling VLA loader.

        Args:
            path: Path to VLA files (can be a directory, file, or glob pattern)
            batch_size: Number of trajectories per batch
            cache_dir: Directory for caching (optional)
            buffer_size: Size of the prefetch buffer
            num_workers: Number of worker processes (-1 for auto, defaults to 2)
            return_type: Return type for data ("numpy" or "torch")
            split: Dataset split ("all", "train", or "val")
        """
        super().__init__(path)
        self.files = self._get_files(path, split)
        self.split = split

        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.return_type = return_type
        self.buffer_size = buffer_size
        self.buffer = mp.Queue(maxsize=buffer_size)

        if num_workers == -1:
            num_workers = 2
        self.num_workers = num_workers
        self.processes = []

        # Shuffle files initially
        random.shuffle(self.files)
        self._start_workers()

    def _get_files(self, path: Text, split: str) -> List[Text]:
        """Get list of VLA files from path with optional train/val split.

        Args:
            path: Path to VLA files
            split: Dataset split ("all", "train", or "val")

        Returns:
            List of VLA file paths
        """
        # Handle split parameter similar to HDF5 loader
        if split and os.path.isdir(path):
            split_path = os.path.join(path, split)
            if os.path.isdir(split_path):
                path = split_path
                logger.info(f"Using split directory: {split_path}")
            else:
                logger.warning(f"Split directory not found: {split_path}, using original path")
        
        ret = []
        if "*" in path:
            ret = glob.glob(path)
        elif os.path.isdir(path):
            ret = glob.glob(os.path.join(path, "*.vla"))
        else:
            ret = [path]

        logger.info(f"Split: {split}, Files: {len(ret)}")
        return ret

    def _read_vla(self, data_path: Text, return_type: Optional[str] = None):
        """Read VLA file and return trajectory data.

        Args:
            data_path: Path to VLA file
            return_type: Return type ("numpy" or "torch")

        Returns:
            Trajectory data
        """
        if return_type is None:
            return_type = self.return_type

        traj = Trajectory(data_path, mode='r')
        ret = traj.load()
        traj.close()

        return ret

    def _worker(self):
        """Worker process that continuously loads random trajectories into the buffer."""
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
        """Start worker processes for prefetching data."""
        for _ in range(self.num_workers):
            p = mp.Process(target=self._worker)
            p.start()
            logger.debug(f"Started worker {p.pid}")
            self.processes.append(p)

    def get_batch(self) -> Optional[List[Any]]:
        """Get a batch of trajectories from the buffer.

        Returns:
            List of trajectories or None if no data available
        """
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
            except Exception:  # mp.queues.Empty
                if all(not p.is_alive() for p in self.processes) and self.buffer.empty():
                    if len(batch) == 0:
                        return None  # No more data available
                    else:
                        break  # Return partial batch

        return batch if batch else None

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self):
        """Get next batch."""
        batch = self.get_batch()
        if batch is None:
            random.shuffle(self.files)
            self._start_workers()
            raise StopIteration
        return batch

    def __len__(self):
        """Get number of files."""
        return len(self.files)

    def peek(self):
        """Peek at a random trajectory."""
        file = random.choice(self.files)
        return self._read_vla(file, return_type="numpy")

    def __del__(self):
        """Clean up worker processes."""
        for p in self.processes:
            p.terminate()
            p.join()


class NonShuffleVLALoader(BaseLoader):
    """Non-shuffling VLA loader that reads trajectories sequentially."""

    def __init__(
        self,
        path: Text,
        batch_size: int = 1,
        cache_dir: Optional[Text] = None,
        num_workers: int = 1,
        return_type: str = "numpy"
    ):
        """Initialize the VLA loader.

        Args:
            path: Path to VLA files (can be a directory, file, or glob pattern)
            batch_size: Number of trajectories per batch
            cache_dir: Directory for caching (optional)
            num_workers: Number of worker processes (not used, for API compatibility)
            return_type: Return type for data ("numpy" or "torch")
        """
        super().__init__(path)
        self.files = self._get_files(path)
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.return_type = return_type
        self.index = 0

    def _get_files(self, path: Text) -> List[Text]:
        """Get list of VLA files from path."""
        ret = []
        if "*" in path:
            ret = glob.glob(path)
        elif os.path.isdir(path):
            ret = glob.glob(os.path.join(path, "*.vla"))
        else:
            ret = [path]
        return ret

    def __iter__(self):
        """Return iterator."""
        self.index = 0
        return self

    def __next__(self):
        """Get next trajectory."""
        if self.index >= len(self.files):
            raise StopIteration

        max_retries = 3
        for attempt in range(max_retries):
            try:
                file_path = self.files[self.index]
                self.index += 1
                return self._read_vla(file_path, return_type=self.return_type)
            except Exception as e:
                logger.error(f"Error reading {file_path} on attempt {attempt + 1}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(f"Failed to read {file_path} after {max_retries} attempts")
                    raise

    def __len__(self):
        """Get number of files."""
        return len(self.files)

    def __getitem__(self, index: int):
        """Get file path at index."""
        return self.files[index]

    def peek(self):
        """Peek at the next trajectory."""
        if self.index < len(self.files):
            file = self.files[self.index]
            return self._read_vla(file, return_type="numpy")
        return None

    def _read_vla(self, data_path: Text, return_type: Optional[str] = None):
        """Read VLA file and return trajectory data."""
        if return_type is None:
            return_type = self.return_type

        traj = Trajectory(data_path, mode='r')
        ret = traj.load()
        traj.close()

        return ret

    def get_batch(self) -> Optional[List[Any]]:
        """Get a batch of trajectories."""
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(self.__next__())
            except StopIteration:
                break
        return batch if batch else None

    def iter_rows(self):
        """Iterate over rows (trajectories)."""
        return iter(self)

    def iter_batches(self):
        """Iterate over batches of trajectories."""
        while True:
            batch = self.get_batch()
            if batch is None:
                break
            yield batch


class VLAIterableDataset(IterableDataset):
    """PyTorch IterableDataset wrapper for VLA data."""

    def __init__(
        self,
        path: Text,
        cache_dir: Optional[Text] = None,
        buffer_size: int = 1000,
        split: str = "train"
    ):
        """Initialize VLA dataset.

        Args:
            path: Path to VLA files
            cache_dir: Cache directory (optional)
            buffer_size: Buffer size for the shuffling loader
        """
        # Use shuffling VLALoader with batch_size=1
        # The DataLoader will handle batching
        self.vla_loader = VLALoader(
            path,
            batch_size=1,
            cache_dir=cache_dir,
            buffer_size=buffer_size,
            split=split
        )

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self):
        """Get next item."""
        batch = self.vla_loader.get_batch()
        if batch is None:
            raise StopIteration
        return batch[0]  # Return a single item, not a batch


def vla_collate_fn(batch):
    """Collate function for VLA data batches."""
    return batch


def get_vla_dataloader(
    path: Text,
    batch_size: int = 1,
    cache_dir: Optional[Text] = None,
    buffer_size: int = 1000,
    num_workers: int = 0,
    split: str = "train"
):
    """Create a PyTorch DataLoader for VLA data.

    Args:
        path: Path to VLA files
        batch_size: Batch size for loading
        cache_dir: Cache directory (optional)
        buffer_size: Buffer size for prefetching
        num_workers: Number of worker processes for DataLoader

    Returns:
        PyTorch DataLoader for VLA data
    """
    dataset = VLAIterableDataset(path, cache_dir, buffer_size, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=vla_collate_fn,
        num_workers=num_workers
    )
