import os

__root_dir__ = os.path.dirname(os.path.abspath(__file__))


from fog_rtx import dataset, episode, feature

all = ["dataset", "feature", "episode"]

import logging

_FORMAT = "%(levelname).1s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=_FORMAT)
logging.root.setLevel(logging.NOTSET)
