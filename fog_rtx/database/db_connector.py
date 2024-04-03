from abc import ABC, abstractmethod
from typing import List

class DatabaseConnector(ABC):
    '''
    abstract class for database connection, such as sqlite, postgres, timescale
    '''
    @abstractmethod
    def add(self, row):
        pass

    @abstractmethod
    def query(self, query:str):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def list_tables(self) -> List[str]:
        pass
