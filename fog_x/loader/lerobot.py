from . import BaseLoader
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class LeRobotLoader(BaseLoader):
    def __init__(self, path, dataset_name, batch_size=1, delta_timestamps=None):
        super(LeRobotLoader, self).__init__(path)
        self.batch_size = batch_size
        self.dataset = LeRobotDataset(root="/mnt/data/fog_x/hf/", repo_id=dataset_name, delta_timestamps=delta_timestamps)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,  # Load one episode at a time
            shuffle=True,
        )
        self.iterator = iter(self.dataloader)
        self.current_episode_index = None

    def __len__(self):
        return len(self.dataset)

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
                    batch = next(self.iterator)
                    episode_index = batch["episode_index"][0].item()

                    from_idx = self.dataset.episode_data_index["from"][episode_index].item()
                    to_idx = self.dataset.episode_data_index["to"][episode_index].item()
                    frames = [_frame_to_numpy(self.dataset[idx]) for idx in range(from_idx, to_idx)]

                    episode.extend(frames)
                    break
                except StopIteration:
                    self.iterator = iter(self.dataloader)
                    if attempt == max_retries - 1:
                        raise StopIteration
                except Exception as e:
                    self.iterator = iter(self.dataloader)
                    if attempt == max_retries - 1:
                        raise e

            batch_of_episodes.append((episode))
            
        
        return batch_of_episodes

    def get_batch(self):
        return next(self)
