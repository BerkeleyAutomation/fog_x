import os

__root_dir__ = os.path.dirname(os.path.abspath(__file__))


from fog_rtx import dataset
from fog_rtx import feature
from fog_rtx import episode
from fog_rtx import db
from fog_rtx import storage

all = [
    "dataset",
    "feature",
    "episode",
    "db",
    "storage"
]

import logging

_FORMAT = '%(levelname).1s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format=_FORMAT)       
logging.root.setLevel(logging.NOTSET)