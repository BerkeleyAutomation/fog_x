
from logging import getLogger

class BaseExporter():
    def __init__(self):
        super(BaseExporter, self).__init__()
        self.logger = getLogger(__name__)

    def export(self, loader, path):
        raise NotImplementedError