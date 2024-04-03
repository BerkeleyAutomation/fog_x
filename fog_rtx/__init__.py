import os

__root_dir__ = os.path.dirname(os.path.abspath(__file__))


from fog_rtx import dataset, db, episode, feature, storage

all = ["dataset", "feature", "episode", "db", "storage"]

import logging

_FORMAT = "%(levelname).1s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=_FORMAT)
logging.root.setLevel(logging.NOTSET)
