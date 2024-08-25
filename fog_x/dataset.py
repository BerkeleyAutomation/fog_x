import os
from typing import Any, Dict, List, Optional, Text

class Dataset:
    def __init__(self, 
                 path: Text,
                 split: Text, 
                 format: Optional[Text] = None):
        """
        init method for Dataset class
        Args:
            paths Text: path-like to the dataset
                it can be a glob pattern or a directory
                if it starts with gs:// it will be treated as a google cloud storage path with rlds format 
                if it ends with .h5 it will be treated as a hdf5 file
                if it ends with .tfrecord it will be treated as a rlds file
                if it ends with .vla it will be treated as a vla file
            split (Text): split of the dataset
            format (Optional[Text]): format of the dataset. Auto-detected if None. Defaults to None.
                we assume that the format is the same for all files in the dataset
        """    
        pass 
    
    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
