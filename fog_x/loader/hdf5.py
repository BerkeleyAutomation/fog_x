

from . import BaseLoader
import numpy as np 
import glob 
import h5py
# flatten the data such that all data starts with root level tree (observation and action)
def _flatten(data, parent_key='', sep='/'):
    items = {}
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items
        
def recursively_read_hdf5_group(group):
    if isinstance(group, h5py.Dataset):
        return np.array(group)
    elif isinstance(group, h5py.Group):
        return {key: recursively_read_hdf5_group(value) for key, value in group.items()}
    else:
        raise TypeError("Unsupported HDF5 group type")
    

class HDF5Loader(BaseLoader):
    def __init__(self, path, split = None):
        super(HDF5Loader, self).__init__(path)
        self.index = 0
        self.files = glob.glob(self.path, recursive=True)

    def __getitem__(self, idx):
        return self._read_hdf5(self.files[idx])

    def _read_hdf5(self, data_path):
        
        with h5py.File(data_path, "r") as f:
            data_unflattened = recursively_read_hdf5_group(f)

        data = {}
        data["observation"] = _flatten(data_unflattened["observation"])
        data["action"] = _flatten(data_unflattened["action"])

        return data

    def __iter__(self):
        return self
    
    def __next__(self):
        # for now naming convention:
        # h/home/kych/datasets/stacking_blocks_trajectories_data/**/trajectory.h5
        if self.index < len(self.files):
            file_path = self.files[self.index]
            self.index += 1
            return self._read_hdf5(file_path)
        raise StopIteration
        
    def __len__(self):
        return len(self.files)