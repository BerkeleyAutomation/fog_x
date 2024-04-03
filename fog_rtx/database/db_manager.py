
from typing import Dict, Any
from fog_rtx.database.db_connector import DatabaseConnector
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    '''
    DatabaseManager class for adding and querying data from the database
    '''
    def __init__(self, db_connector: DatabaseConnector, dataset_name: str):
        self.db_connector = db_connector
        self.dataset_name = dataset_name
        self._initialize_dataset(dataset_name)

    def _initialize_dataset(self, dataset_name: str):
        tables = self.db_connector.list_tables()
        logger.info(f"Tables in database: {tables}")
        if tables and dataset_name in tables:
            logger.info("Database is not empty and already initialized")
        else: 
            self.db_connector.create_table(dataset_name, {"Episode_Description": "TEXT"})
            logger.info("Database initialized")  

    def _initialize_episode(self, metadata: Dict[str, Any]):
        # TODO: placeholder, add other metadata to the database 
        episode_description = metadata["description"]
        # insert episode information to the database
        self.db_connector.insert_data(self.dataset_name, {"Episode_Description": episode_description})

    def add(self, feature, value, timestamp):
        # self.database.add(row)
        pass

    def query(self, key):
        return self.database.query(key)

    def close(self):
        self.database.close()
