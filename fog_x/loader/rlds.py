from . import BaseLoader
import numpy as np

class RLDSLoader(BaseLoader):
    def __init__(self, path, split, batch_size=1, shuffle_buffer=50):
        super(RLDSLoader, self).__init__(path)
        
        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("Please install tensorflow and tensorflow_datasets to use rlds loader")

        self.batch_size = batch_size
        builder = tfds.builder_from_directory(path)
        self.ds = builder.as_dataset(split)
        self.length = len(self.ds)
        self.ds = self.ds.shuffle(shuffle_buffer)
        self.ds = self.ds.repeat()
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
        data = []
        for b in batch:
            data.append(self._convert_batch_to_numpy(b))
        return data


    def _convert_batch_to_numpy(self, batch):
        import tensorflow as tf

        def to_numpy(step_data):
            step = {}
            for key, val in step_data.items():
                if key == "observation":
                    step["observation"] = {obs_key: np.array(obs_val) for obs_key, obs_val in val.items()}
                elif key == "action":
                    step["action"] = {act_key: np.array(act_val) for act_key, act_val in val.items()}
                else:
                    step[key] = np.array(val)
            return step

        batch = to_numpy(batch)
        return batch

    def __next__(self):
        if self.index >= self.length:
            self.index = 0
            raise StopIteration
        return self.get_batch()

    def __getitem__(self, idx):
        batch = next(iter(self.ds.skip(idx).take(1)))
        return self._convert_batch_to_numpy(batch)
