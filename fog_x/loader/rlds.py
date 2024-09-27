from . import BaseLoader
import numpy as np


class RLDSLoader(BaseLoader):
    def __init__(self, path, split, batch_size=1, shuffle_buffer=10, shuffling = True):
        super(RLDSLoader, self).__init__(path)

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "Please install tensorflow and tensorflow_datasets to use rlds loader"
            )

        self.batch_size = batch_size
        builder = tfds.builder_from_directory(path)
        self.ds = builder.as_dataset(split)
        self.length = len(self.ds)
        self.shuffling = shuffling
        if shuffling:
            self.ds = self.ds.repeat()
            self.ds = self.ds.shuffle(shuffle_buffer)
        self.iterator = iter(self.ds)

        self.split = split
        self.index = 0

    def __len__(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("Please install tensorflow to use rlds loader")

        return self.length

    def __iter__(self):
        return self

    def get_batch(self):
        batch = self.ds.take(self.batch_size)
        self.index += self.batch_size
        if not self.shuffling and self.index >= self.length:
            raise StopIteration
        data = []
        for b in batch:
            data.append(self._convert_traj_to_numpy(b))
        return data

    def _convert_traj_to_numpy(self, traj):
        import tensorflow as tf

        def to_numpy(step_data):
            step = {}
            for key in step_data:
                val = step_data[key]
                if isinstance(val, dict):
                    step[key] = {k: np.array(v) for k, v in val.items()}
                else:
                    step[key] = np.array(val)
            return step

        trajectory = []
        for step in traj["steps"]:
            trajectory.append(to_numpy(step))
        return trajectory

    def __next__(self):
        data = [self._convert_traj_to_numpy(next(self.iterator))]
        self.index += 1
        if self.index >= self.length:
            raise StopIteration
        return data

    def __getitem__(self, idx):
        batch = next(iter(self.ds.skip(idx).take(1)))
        return self._convert_traj_to_numpy(batch)
    
class RLDSLoader_ByFrame(BaseLoader):
    def __init__(self, path, split, batch_size=1, shuffle_buffer=10, shuffling = True, slice_length=16):
        super(RLDSLoader_ByFrame, self).__init__(path)

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "Please install tensorflow and tensorflow_datasets to use rlds loader"
            )

        self.batch_size = batch_size
        builder = tfds.builder_from_directory(path)
        self.ds = builder.as_dataset(split)
        self.length = len(self.ds)
        self.shuffling = shuffling
        if shuffling:
            self.ds = self.ds.repeat()
            self.ds = self.ds.shuffle(shuffle_buffer)
        self.slice_length = slice_length
        self.iterator = iter(self.ds)

        self.split = split
        self.index = 0

    def __len__(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("Please install tensorflow to use rlds loader")

        return self.length

    def __iter__(self):
        return self

    def get_batch(self):
        batch = self.ds.take(self.batch_size)
        self.index += self.batch_size
        if not self.shuffling and self.index >= self.length:
            raise StopIteration
        data = []
        for b in batch:
            data.append(self._convert_traj_to_numpy(b))
        return data

    def _convert_traj_to_numpy(self, traj):
        import tensorflow as tf

        def to_numpy(step_data):
            step = {}
            for key in step_data:
                val = step_data[key]
                if isinstance(val, dict):
                    step[key] = {k: np.array(v) for k, v in val.items()}
                else:
                    step[key] = np.array(val)
            return step

        # Random step / frame slicing
        trajectory = []
        num_frames = len(traj["steps"])
        if num_frames >= self.slice_length:
            random_from = np.random.randint(0, num_frames - self.slice_length + 1)
            # random_to = random_from + self.slice_length
            trajs = traj["steps"].skip(random_from).take(self.slice_length)
        else:
            # random_from = 0
            # random_to = num_frames
            trajs = traj["steps"]
        for step in trajs:
            trajectory.append(to_numpy(step))
        return trajectory

    def __next__(self):
        data = [self._convert_traj_to_numpy(next(self.iterator))]
        self.index += 1
        if self.index >= self.length:
            raise StopIteration
        return data

    def __getitem__(self, idx):
        batch = next(iter(self.ds.skip(idx).take(1)))
        return self._convert_traj_to_numpy(batch)