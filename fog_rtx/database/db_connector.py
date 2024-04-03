from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DatabaseConnector(ABC):
    """
    abstract class for database connection, such as sqlite, postgres, timescale
    """

    @abstractmethod
    def add(self, row):
        pass

    @abstractmethod
    def query(self, query: str):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def list_tables(self) -> List[str]:
        pass

    @abstractmethod
    def create_table(self, table_name: str, columns: Dict[str, str]):
        """
        columns: a dictionary of column names and their SQL types
        """
        pass

    @abstractmethod
    def insert_data(self, table_name: str, data: Any) -> int:
        """
        return the index of the inserted data
        """
        pass
