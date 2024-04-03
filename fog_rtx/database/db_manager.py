
from fog_rtx.database.db_connector import DatabaseConnector
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    '''
    DatabaseManager class for adding and querying data from the database
    '''
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self._initialize_database()

    def _initialize_database(self):
        tables = self.db_connector.list_tables()
        logger.info(f"Tables in database: {tables}")
        if tables and "Metadata" in tables:
            logger.info("Database is not empty and already initialized")
        else: 
            self.db_connector.create_table("Metadata", {"Episode_Description": "TEXT"})
            logger.info("Database initialized")

    def add(self, feature, value):
        # self.database.add(row)
        pass

    def query(self, key):
        return self.database.query(key)

    def close(self):
        self.database.close()
