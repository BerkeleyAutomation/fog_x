from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def add(self, row):
        pass

    @abstractmethod
    def query(self, key):
        pass

    @abstractmethod
    def close(self):
        pass
