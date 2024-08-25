from logging import getLogger


class BaseLoader():
    def __init__(self, 
                 path, 
                 split = None):
        super(BaseLoader, self).__init__()
        self.logger = getLogger(__name__)
        self.path = path
        self.split = split

    # def get_schema(self) -> Schema:
    #     raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    def __iter___(self):
        raise NotImplementedError
