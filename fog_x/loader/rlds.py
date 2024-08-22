

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

        self.split = split
        self.index = 0

    def __len__(self):
        return len(self.ds)
    
    def __iter__(self):
        return self
    
    def __next__(self):

        if self.index < len(self):
            self.index += 1
            nest_ds = self.ds.__iter__()
            traj = list(nest_ds)[0]["steps"]
            data = []

            for step_data in traj:
                step = {}
                for key, val in step_data.items():

                    if key == "observation":
                        step["observation"] = {}
                        for obs_key, obs_val in val.items():
                            step["observation"][obs_key] = np.array(obs_val)

                    elif key == "action":
                        step["action"] = {}
                        for act_key, act_val in val.items():
                            step["action"][act_key] = np.array(act_val)
                    else:
                        step[key] = np.array(val)

                data.append(step)
            return data
        else:
            self.index = 0
            raise StopIteration
        