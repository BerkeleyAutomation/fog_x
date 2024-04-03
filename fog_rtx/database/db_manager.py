
from fog_rtx.database.db_connector import DatabaseConnector
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    '''
    DatabaseManager class for adding and querying data from the database
    '''
    def __init__(self, database: DatabaseConnector):
        self.db_connector = db_connector
        self._initialize_database()

    def _initialize_database(self):
        tables = self.db_connector.list_tables()
        if 

    def add(self, feature, value):
        # self.database.add(row)
        pass

    def query(self, key):
        return self.database.query(key)

    def close(self):
        self.database.close()
