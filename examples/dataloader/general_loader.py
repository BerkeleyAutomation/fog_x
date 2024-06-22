import fog_x 
import os
from logging import getLogger
import numpy as np

class BaseLoader():
    def __init__(self, data_path):
        super(BaseLoader, self).__init__()
        self.data_dir = data_path
        self.logger = getLogger(__name__)


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __iter___(self):
        raise NotImplementedError
    
class RTXLoader(BaseLoader):
    def __init__(self, data_path):
        super(RTXLoader, self).__init__(data_path)
        import tensorflow_datasets as tfds

        builder = tfds.builder_from_directory(data_path)

        self.ds = builder.as_dataset(split="train[:50]").shuffle(50)
        # https://www.determined.ai/blog/tf-dataset-the-bad-parts
        # data_source = builder.as_data_source()
        # print(data_source)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return idx
    
    def __iter__(self):
        return self.ds.__iter__()

dataset = RTXLoader(os.path.expanduser("~/datasets/berkeley_autolab_ur5/0.1.0"))
# RTXLoader("gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0")


class BaseExporter():
    def __init__(self):
        super(BaseExporter, self).__init__()
        self.logger = getLogger(__name__)

    def export(self, loader: BaseLoader, output_path: str):
        raise NotImplementedError

class MKVExporter(BaseExporter):
    def __init__(self):
        super(MKVExporter, self).__init__()

    def export(self, loader: BaseLoader, output_path: str):
        for i in loader:
            np.array(loader[i])

exporter = MKVExporter()
exporter.export(dataset, "output.mkv")