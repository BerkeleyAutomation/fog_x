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
                    try:
                        from_idx = self.dataset.episode_data_index["from"][self.episode_index].item()
                        to_idx = self.dataset.episode_data_index["to"][self.episode_index].item()
                    except Exception as e:
                        self.episode_index = 0
                        continue
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


class LeRobotLoader_ByFrame(BaseLoader):
    def __init__(self, path, dataset_name, batch_size=1, delta_timestamps=None, slice_length=16):
        super(LeRobotLoader_ByFrame, self).__init__(path)
        self.batch_size = batch_size
        self.dataset = LeRobotDataset(root="/mnt/data/fog_x/hf/", repo_id=dataset_name, delta_timestamps=delta_timestamps)
        self.slice_length = slice_length

    def __len__(self):
        return len(self.dataset.episode_data_index["from"])

    def __iter__(self):
        return self

    def __next__(self):
        max_retries = 10
        batch_of_episodes = []

        def _frame_to_numpy(frame):
            return {k: np.array(v) for k, v in frame.items()}
        for _ in range(self.batch_size):
            episode = []
            for attempt in range(max_retries):
                try:
                    # repeat
                    self.episode_index = np.random.randint(0, len(self.dataset))
                    try:
                        from_idx = self.dataset.episode_data_index["from"][self.episode_index].item()
                        to_idx = self.dataset.episode_data_index["to"][self.episode_index].item()
                    except Exception as e:
                        continue
                    
                    # Randomly select random_frames from episode
                    episode_length = to_idx - from_idx
                    if episode_length <= self.slice_length:
                        random_from = from_idx
                        random_to = to_idx
                    else:
                        random_from = np.random.randint(from_idx, to_idx - self.slice_length)
                        random_to = random_from + self.slice_length
                    frames = [_frame_to_numpy(self.dataset[idx]) for idx in range(random_from, random_to)]
                    episode.extend(frames)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        continue

            batch_of_episodes.append((episode))
            
        
        return batch_of_episodes

    def get_batch(self):
        return next(self)
