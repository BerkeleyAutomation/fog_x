

from . import BaseLoader
import os
import sys
import numpy as np

class RLDSLoader(BaseLoader):
    def __init__(self, path, split):
        super(RLDSLoader, self).__init__(path)
        
        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("Please install tensorflow and tensorflow_datasets to use rlds loader")

        builder = tfds.builder_from_directory(path)
        self.ds = builder.as_dataset(split)
        self.iterator = iter(self.ds)

        self.split = split
        self.index = 0

    def __len__(self):
        return tf.data.experimental.cardinality(self.ds).numpy()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            nest_ds = next(self.iterator)
            traj = nest_ds["steps"]
            data = []

            for step_data in traj:
                step = {}
                for key, val in step_data.items():
                    if key == "observation":
                        step["observation"] = {obs_key: np.array(obs_val) for obs_key, obs_val in val.items()}
                    elif key == "action":
                        step["action"] = {act_key: np.array(act_val) for act_key, act_val in val.items()}
                    else:
                        step[key] = np.array(val)
                data.append(step)
            return data
        except StopIteration:
            self.index = 0
            self.iterator = iter(self.ds)
            raise StopIteration