
from typing import Dict, Any, List
from fog_rtx.database.db_connector import DatabaseConnector
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    '''
    DatabaseManager class for adding and querying data from the database
    '''
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.dataset_name = None 
        self.current_episode_id = None 
        self.features = []

    def initialize_dataset(self, dataset_name: str, features: List[str]):
        self.dataset_name = dataset_name
        self.features = features
        tables = self.db_connector.list_tables()
        logger.info(f"Tables in database: {tables}")
        if tables and dataset_name in tables:
            logger.info("Database is not empty and already initialized")
        else: 
            self.db_connector.create_table(dataset_name, {"Episode_Description": "TEXT"})
            logger.info("Database initialized")  
        

    def initialize_episode(self, 
                            metadata: Dict[str, Any], 
                            ) -> int:
        # TODO: placeholder, add other metadata to the database 
        episode_description = metadata["description"]
        # insert episode information to the database
        self.current_episode_id = self.db_connector.insert_data(self.dataset_name, {"Episode_Description": episode_description})

        # create tables for each feature
        for feature in self.features:
            self._initialize_feature(feature)

        return self.current_episode_id

    def add(self, feature, value, timestamp):
        if feature not in self.features:
            logger.warning(f"Feature {feature} not in the list of features")
            self._initialize_feature(feature)
            self.features.append(feature)

        # insert data into the table
        self.db_connector.insert_data(self._get_feature_table_name(feature), {"Timestamp": timestamp, "Value": value})
    
    def _initialize_feature(self, feature: str):
        # create a table for the feature
        # TODO: need to make the timestamp type as TIMESTAMPTZ
        self.db_connector.create_table(self._get_feature_table_name(feature), {"Timestamp": "INT", "Value": "TEXT"})
    
    def _get_feature_table_name(self, feature):
        if self.dataset_name is None:
            raise ValueError("Dataset not initialized")
        if self.current_episode_id is None:
            raise ValueError("Episode not initialized")
        table_name = f"{self.dataset_name}_{self.current_episode_id}_{feature}"
        return table_name


    def query(self, key):
        return self.database.query(key)

    def close(self):
        self.database.close()
