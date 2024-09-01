from . import BaseLoader
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class LeRobotLoader(BaseLoader):
    def __init__(self, path, dataset_name, batch_size=1, delta_timestamps=None):
        super(LeRobotLoader, self).__init__(path)
        self.batch_size = batch_size
        self.dataset = LeRobotDataset(root = "/mnt/data/fog_x/hf/", repo_id =dataset_name, delta_timestamps=delta_timestamps)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.iterator = iter(self.dataloader)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                batch = next(self.iterator)
                break
            except StopIteration:
                self.iterator = iter(self.dataloader)
                if attempt == max_retries - 1:
                    raise StopIteration
            except Exception as e:
                # print(f"Error in __next__ (attempt {attempt + 1}/{max_retries}): {e}")
                self.iterator = iter(self.dataloader)
                if attempt == max_retries - 1:
                    raise e
        return self._convert_batch_to_numpy(batch)

    def _convert_batch_to_numpy(self, batch):
        numpy_batch = []
        for i in range(len(next(iter(batch.values())))):
            trajectory = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    trajectory[key] = value[i].numpy()
                elif isinstance(value, dict):
                    trajectory[key] = self._convert_batch_to_numpy({k: v[i] for k, v in value.items()})
                else:
                    trajectory[key] = value[i]
            numpy_batch.append(trajectory)
        return numpy_batch

    def get_batch(self):
        return next(self)
