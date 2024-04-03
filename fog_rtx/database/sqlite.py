from fog_rtx.database import Database


class SQLite(Database):
    def __init__(self, path: str):
        self.path = path

    def add(self, key, value):
        pass

    def query(self, key):
        pass

    def close(self):
        pass
