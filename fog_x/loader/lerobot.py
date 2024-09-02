from . import BaseLoader
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class LeRobotLoader(BaseLoader):
    def __init__(self, path, dataset_name, batch_size=1, delta_timestamps=None):
        super(LeRobotLoader, self).__init__(path)
        self.batch_size = batch_size
        self.dataset = LeRobotDataset(root="/mnt/data/fog_x/hf/", repo_id=dataset_name, delta_timestamps=delta_timestamps)
        self.episode_index = 0

    def __len__(self):
        return len(self.dataset.episode_data_index["from"])

    def __iter__(self):
        return self

    def __next__(self):
        max_retries = 3
        batch_of_episodes = []

        def _frame_to_numpy(frame):
            return {k: np.array(v) for k, v in frame.items()}
        for _ in range(self.batch_size):
            episode = []
            for attempt in range(max_retries):
                try:
                    # repeat
                    if self.episode_index >= len(self.dataset):
                        self.episode_index = 0
                    from_idx = self.dataset.episode_data_index["from"][self.episode_index].item()
                    to_idx = self.dataset.episode_data_index["to"][self.episode_index].item()
                    frames = [_frame_to_numpy(self.dataset[idx]) for idx in range(from_idx, to_idx)]
                    episode.extend(frames)
                    self.episode_index += 1
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    self.episode_index += 1


            batch_of_episodes.append((episode))
            
        
        return batch_of_episodes

    def get_batch(self):
        return next(self)
