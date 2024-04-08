import logging
from typing import Any, Dict, List, Optional

from fog_rtx.database.polars_connector import PolarsConnector
from fog_rtx.feature import FeatureType
from ast import literal_eval
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    DatabaseManager class for adding and querying data from the database
    """

    def __init__(self, 
                 episode_info_connector: PolarsConnector,
                 step_data_connector: PolarsConnector):
        self.episode_info_connector = episode_info_connector
        self.step_data_connector = step_data_connector
        self.dataset_name: Optional[str] = None
        self.current_episode_id: Optional[int] = None
        self.features: Dict[str, FeatureType] = {}
        self.current_episode_id = 0

    def initialize_dataset(
        self, dataset_name: str, features: Dict[str, FeatureType]
    ):
        self.dataset_name = dataset_name
        self.features = features
        self.episode_info_connector.load_tables([self.dataset_name])

        # self.step_data_connector.load_tables()
        # tables = self.episode_info_connector.list_tables()

        episode_info_table = None 
        episode_info_table = self.get_episode_info_table()
        logger.info(f"Episode Info Table: {episode_info_table}")
        if episode_info_table is not None:
            for row in episode_info_table.iter_rows(named=True):
                if None in row.values():
                    continue 
                for key, value in row.items():
                    if key.endswith("type"):
                        feature_name = key.replace("feature_", "").replace("_type", "")
                        feature_type = FeatureType(
                            dtype=value, shape=literal_eval(row[f"feature_{feature_name}_shape"])
                        )
                        self.features[feature_name] = feature_type
                        logger.info(
                            f"Loaded Features: {feature_name} with type {feature_type}"
                        )
                break
            # if tables and dataset_name in tables:
            #     logger.info("Database is not empty and already initialized")
            #     # get features from the table and update the features
            #     metadata = self.get_metadata_table()
            #     for _, row in metadata.iterrows():
            #         feature_type = FeatureType(
            #             dtype=row["Type"], shape=row["Shape"]
            #         )
            #         self.features[row["Feature"]] = feature_type
            #     logger.info(
            #         f"Loaded Features: {self.features} with type {feature_type}"
            #     )
        else:
            self.episode_info_connector.create_table(
                dataset_name,
            )
            self.episode_info_connector.add_column(
                dataset_name,
                "episode_id",
                "int64",
            )
            self.episode_info_connector.add_column(
                dataset_name,
                "Compacted",
                "bool",
            )
            logger.info("Database initialized")

    def initialize_episode(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if self.dataset_name is None:
            raise ValueError("Dataset not initialized")
        if metadata is None:
            metadata = {}
        logger.info(
            f"Initializing episode for dataset {self.dataset_name} with metadata {metadata}"
        )

        metadata["episode_id"] = self.current_episode_id
        metadata["Compacted"] = False

        for metadata_key in metadata.keys():
            logger.info(f"Adding metadata key {metadata_key} to the database")
            self.episode_info_connector.add_column(
                self.dataset_name,
                metadata_key,
                "str",  # TODO: support more types
            )

        # insert episode information to the database
        self.current_episode_id = self.episode_info_connector.insert_data(
            self.dataset_name,
            metadata,
        )

        # create tables for each feature
        for feature_name, feature_type in self.features.items():
            self._initialize_feature(feature_name, feature_type)

        return self.current_episode_id


    def _initialize_feature(
        self, feature_name: str, feature_type: FeatureType
    ):
        # create a table for the feature
        # TODO: need to make the timestamp type as TIMESTAMPTZ
        self.step_data_connector.create_table(
            self._get_feature_table_name(feature_name),
        )
        # {"Timestamp": "int64", feature_name: feature_type.to_sql_type()}

        self.step_data_connector.add_column(
            self._get_feature_table_name(feature_name),
            "episode_id",
            "int64",
        )
        self.step_data_connector.add_column(
            self._get_feature_table_name(feature_name),
            "Timestamp",
            "int64",
        )
        logger.info(
            f"Adding feature {feature_name} to the database with type {feature_type.to_pld_storage_type()}"
        )
        self.step_data_connector.add_column(
            self._get_feature_table_name(feature_name),
            feature_name,
            feature_type.to_pld_storage_type(),  
        )

        if feature_type is None:
            logger.error(f"Feature type not provided for {feature_name}")
            self.features[feature_name] = FeatureType()
        else:
            self.features[feature_name] = feature_type

        # update dataset metadata with the feature information
        self.episode_info_connector.add_column(
            self.dataset_name,
            f"feature_{feature_name}_type",
            "str",
        )
        self.episode_info_connector.add_column(
            self.dataset_name,
            f"feature_{feature_name}_shape",
            "str",
        )
        self.episode_info_connector.update_data(
            table_name=self.dataset_name,
            index=self.current_episode_id,
            data={
                f"feature_{feature_name}_type": str(
                    self.features[feature_name].dtype
                ),
                f"feature_{feature_name}_shape": str(
                    self.features[feature_name].shape
                ),
            },
        )


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
            if feature_type is None:

                feature_type = FeatureType(data=value)
                logger.warn(
                    f"feature type not provided, inferring from data type {feature_type}"
                )
            self._initialize_feature(feature_name, feature_type)

        # insert data into the table
        self.step_data_connector.insert_data(
            self._get_feature_table_name(feature_name),
            {"episode_id": self.current_episode_id,
             "Timestamp": timestamp, 
             feature_name: value},
        )

    def compact(self):
        # create a table for the compacted data
        # iterate through all the features and get the data
        table_names = [
            self._get_feature_table_name(feature_name)
            for feature_name in self.features.keys()
        ]
        logger.info(f"Compacting and Merging tables: {table_names}")

        self.step_data_connector.merge_tables_with_timestamp(
            table_names,
            f"{self.dataset_name}_{self.current_episode_id}_compacted",
        )

        # update the metadata field marking the episode as compacted
        self.episode_info_connector.update_data(
            self.dataset_name,
            self.current_episode_id,
            {"Compacted": True},
        )

    # def get_table(
    #     self, table_name: Optional[str] = None, format: str = "pandas"
    # ):
    #     if table_name is None:
    #         table_name = self.dataset_name
    #     if table_name is None:
    #         raise ValueError("Dataset name not provided")
    #     return self.db_connector.select_table(table_name)

    def get_episode_info_table(self):
        return self.episode_info_connector.select_table(self.dataset_name)

    def get_step_table(self, episode_id):
        if episode_id is None:
            raise ValueError("Episode not initialized")
        return self.step_data_connector.select_table(
            f"{self.dataset_name}_{episode_id}_compacted",
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

    def close(self):
        self.compact()
        self.step_data_connector.save_tables(
            [f"{self.dataset_name}_{self.current_episode_id}_compacted"],
        )
        self.episode_info_connector.save_tables(
            [f"{self.dataset_name}"],
        )
