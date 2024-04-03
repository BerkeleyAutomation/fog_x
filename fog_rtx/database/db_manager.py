import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import Integer, String  # type: ignore

from fog_rtx.database.db_connector import DatabaseConnector
from fog_rtx.feature import FeatureType

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    DatabaseManager class for adding and querying data from the database
    """

    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.dataset_name: Optional[str] = None
        self.current_episode_id: Optional[int] = None
        self.features: Dict[str, FeatureType] = {}

    def initialize_dataset(
        self, dataset_name: str, features: Dict[str, FeatureType]
    ):
        self.dataset_name = dataset_name
        self.features = features
        tables = self.db_connector.list_tables()
        logger.info(f"Tables in database: {tables}")
        if tables and dataset_name in tables:
            logger.info("Database is not empty and already initialized")
        else:
            self.db_connector.create_table(
                dataset_name, {"Episode_Description": String}
            )
            logger.info("Database initialized")

    def initialize_episode(
        self,
        metadata: Dict[str, Any],
    ) -> int:
        if self.dataset_name is None:
            raise ValueError("Dataset not initialized")
        # TODO: placeholder, add other metadata to the database
        episode_description = metadata["description"]
        # insert episode information to the database
        self.current_episode_id = self.db_connector.insert_data(
            self.dataset_name, {"Episode_Description": episode_description}
        )

        # create tables for each feature
        for feature_name in self.features.keys():
            self._initialize_feature(feature_name)

        return self.current_episode_id

    def add(
        self,
        feature_name: str,
        value: Any,
        timestamp: int,
        feature_type: Optional[FeatureType] = None,
    ):
        if feature_name not in self.features.keys():
            logger.warning(
                f"Feature {feature_name} not in the list of features"
            )
            self._initialize_feature(feature_name)
            if feature_type is None:
                logger.error(f"Feature type not provided for {feature_name}")
            else:
                self.features[feature_name] = feature_type

        # insert data into the table
        self.db_connector.insert_data(
            self._get_feature_table_name(feature_name),
            {"Timestamp": timestamp, feature_name: value},
        )

    def compact(self):
        # create a table for the compacted data
        # iterate through all the features and get the data
        table_names = [
            self._get_feature_table_name(feature_name)
            for feature_name in self.features
        ]

    def _initialize_feature(self, feature_name: str):
        # create a table for the feature
        # TODO: need to make the timestamp type as TIMESTAMPTZ
        self.db_connector.create_table(
            self._get_feature_table_name(feature_name),
            {"Timestamp": Integer, feature_name: String},
        )

    def _get_feature_table_name(self, feature_name):
        if self.dataset_name is None:
            raise ValueError("Dataset not initialized")
        if self.current_episode_id is None:
            raise ValueError("Episode not initialized")
        table_name = (
            f"{self.dataset_name}_{self.current_episode_id}_{feature_name}"
        )
        return table_name

    def query(self, key):
        return self.database.query(key)

    def close(self):
        self.database.close()
