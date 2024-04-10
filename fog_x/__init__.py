import os

__root_dir__ = os.path.dirname(os.path.abspath(__file__))


from fog_x import dataset, episode, feature
from fog_x.dataset import Dataset

all = ["dataset", "feature", "episode", "Dataset"]

import logging

_FORMAT = "%(levelname).1s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=_FORMAT)
logging.root.setLevel(logging.INFO)
