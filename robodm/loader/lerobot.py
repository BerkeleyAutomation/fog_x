"""LeRobot dataset loader for robodm."""

import logging
from typing import Optional

import numpy as np

from . import BaseLoader

logger = logging.getLogger(__name__)


class LeRobotLoader(BaseLoader):
    """Loader for LeRobot datasets (HuggingFace format)."""

    def __init__(
        self,
        path: str,
        dataset_name: str,
        batch_size: int = 1,
        delta_timestamps: Optional[dict] = None,
        root: Optional[str] = None
    ):
        """Initialize LeRobot loader.

        Args:
            path: Path to the dataset directory (used as root if root is None)
            dataset_name: Name/repo_id of the LeRobot dataset
            batch_size: Number of episodes per batch
            delta_timestamps: Optional delta timestamps for the dataset
            root: Root directory for LeRobot datasets (defaults to path if None)
        """
        super().__init__(path)

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise ImportError(
                "Please install lerobot to use the LeRobot loader: pip install lerobot"
            )

        self.batch_size = batch_size

        # Use path as root if root is not provided
        # Note: LeRobot expects root to point to the dataset directory itself,
        # not the parent directory. So we need to append dataset_name to path.
        if root is None:
            import os
            root = os.path.join(path, dataset_name)

        # Initialize LeRobot dataset
        self.dataset = LeRobotDataset(
            repo_id=dataset_name,
            root=root,
            delta_timestamps=delta_timestamps
        )
        self.episode_index = 0

    def __len__(self):
        """Return number of episodes in the dataset."""
        return len(self.dataset.episode_data_index["from"])

    def __iter__(self):
        """Return iterator."""
        self.episode_index = 0
        return self

    def __next__(self):
        """Get next batch of episodes."""
        max_retries = 3
        batch_of_episodes = []

        def _frame_to_numpy(frame):
            """Convert a frame to numpy format."""
            return {k: np.array(v) if not isinstance(v, np.ndarray) else v
                    for k, v in frame.items()}

        for _ in range(self.batch_size):
            episode = []
            for attempt in range(max_retries):
                try:
                    # Wrap around if we reach the end
                    if self.episode_index >= len(self):
                        self.episode_index = 0
                        raise StopIteration

                    try:
                        from_idx = self.dataset.episode_data_index["from"][
                            self.episode_index
                        ].item()
                        to_idx = self.dataset.episode_data_index["to"][
                            self.episode_index
                        ].item()
                    except Exception as e:
                        logger.warning(f"Error getting episode indices: {e}")
                        self.episode_index = 0
                        continue

                    # Load all frames for this episode
                    frames = [
                        _frame_to_numpy(self.dataset[idx])
                        for idx in range(int(from_idx), int(to_idx))
                    ]
                    episode.extend(frames)
                    self.episode_index += 1
                    break

                except StopIteration:
                    raise
                except Exception as e:
                    logger.error(f"Error loading episode {self.episode_index}: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    self.episode_index += 1

            batch_of_episodes.append(episode)

        return batch_of_episodes

    def get_batch(self):
        """Get a batch of episodes."""
        return next(self)
