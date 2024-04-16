import logging
from ast import literal_eval
from typing import Any, Dict, List, Optional

from fog_x.database.polars_connector import PolarsConnector
from fog_x.feature import FeatureType

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    DatabaseManager class for adding and querying data from the database
    """

    def __init__(
        self,
        episode_info_connector: PolarsConnector,
        step_data_connector: PolarsConnector,
        required_stats: List[str] = ["count"],
    ):
        self.episode_info_connector = episode_info_connector
        self.step_data_connector = step_data_connector
        self.dataset_name: Optional[str] = None
        self.current_episode_id: Optional[int] = None
        self.features: Dict[str, FeatureType] = {}
        self.current_episode_id = -1
        self.episode_metadata = {}
        self.required_stats = required_stats

    def initialize_dataset(
        self, dataset_name: str, features: Dict[str, FeatureType]
    ):
        self.dataset_name = dataset_name
        self.features = features
        self.episode_info_connector.load_tables([self.dataset_name])

        # tables = self.episode_info_connector.list_tables()

        episode_info_table = None
        episode_info_table = self.get_episode_info_table()

        logger.debug(f"Episode Info Table: {episode_info_table}")
        if episode_info_table is not None:
            self.current_episode_id = len(episode_info_table)
            for row in episode_info_table.iter_rows(named=True):
                if None in row.values():
                    continue
                for key, value in row.items():
                    if key.endswith("type"):
                        feature_name = key.replace("feature_", "").replace(
                            "_type", ""
                        )
                        feature_type = FeatureType(
                            dtype=value,
                            shape=literal_eval(
                                row[f"feature_{feature_name}_shape"]
                            ),
                        )
                        self.features[feature_name] = feature_type
                        logger.debug(
                            f"Loaded Features: {feature_name} with type {feature_type}"
                        )
                break

            self.step_data_connector.load_tables(
                [episode_id for episode_id in range(len(episode_info_table))],
                [
                    f"{self.dataset_name}_{episode_id}"
                    for episode_id in range(len(episode_info_table))
                ],
            )
            # if tables and dataset_name in tables:
            #     logger.debug("Database is not empty and already initialized")
            #     # get features from the table and update the features
            #     metadata = self.get_metadata_table()
            #     for _, row in metadata.iterrows():
            #         feature_type = FeatureType(
            #             dtype=row["Type"], shape=row["Shape"]
            #         )
            #         self.features[row["Feature"]] = feature_type
            #     logger.debug(
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
                "Finished",
                "bool",
            )
            self.current_episode_id = 0
            logger.debug("Database initialized")

    def initialize_episode(
        self,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if self.dataset_name is None:
            raise ValueError("Dataset not initialized")
        if additional_metadata is None:
            additional_metadata = {}
        for key, value in additional_metadata.items():
            self.episode_metadata[key] = value

        self.episode_metadata["episode_id"] = self.current_episode_id
        self.episode_metadata["Finished"] = False
        logger.debug(
            f"Initializing episode for dataset {self.dataset_name} with metadata {self.episode_metadata}"
        )
        for metadata_key in self.episode_metadata.keys():
            logger.debug(f"Adding metadata key {metadata_key} to the database")
            self.episode_info_connector.add_column(
                self.dataset_name,
                metadata_key,
                "str",  # TODO: support more types
            )

        # insert episode information to the database
        self.episode_info_connector.insert_data(
            self.dataset_name,
            self.episode_metadata,
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
        logger.debug(
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
        for stat in self.required_stats:
            self.episode_info_connector.add_column(
                self.dataset_name,
                f"{feature_name}_{stat}",
                "float64",
            )

    def add(
        self,
        feature_name: str,
        value: Any,
        timestamp: int,
        feature_type: Optional[FeatureType] = None,
        metadata_only = False,
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

        if not metadata_only:
            # insert data into the table
            self.step_data_connector.insert_data(
                self._get_feature_table_name(feature_name),
                {
                    "episode_id": self.current_episode_id,
                    "Timestamp": timestamp,
                    feature_name: value,
                },
            )

    def compact(self):
        # create a table for the compacted data
        # iterate through all the features and get the data
        table_names = [
            self._get_feature_table_name(feature_name)
            for feature_name in self.features.keys()
        ]
        logger.debug(f"Compacting and Merging tables: {table_names}")

        self.step_data_connector.merge_tables_with_timestamp(
            table_names,
            f"{self.dataset_name}_{self.current_episode_id}",
            clear_feature_tables = True,
        )

        self.step_data_connector.remove_tables(table_names)

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
            f"{self.dataset_name}_{episode_id}",
        )

    def get_step_table_all(self, reload=False):
        return self.step_data_connector.get_dataset_table(reload)

    def _get_feature_table_name(self, feature_name):
        if self.dataset_name is None:
            raise ValueError("Dataset not initialized")
        if self.current_episode_id is None:
            raise ValueError("Episode not initialized")
        table_name = (
            f"{self.dataset_name}_{self.current_episode_id}_{feature_name}"
        )
        return table_name

    def close(self,  save_data=True, save_metadata=True, additional_metadata = None):

        if additional_metadata is None:
            additional_metadata = {}
        logger.info(f"Closing the episode with metadata {additional_metadata}")
        additional_metadata["Finished"] = True
        if save_data:
            self.compact()
            compacted_table = self.step_data_connector.select_table(
                f"{self.dataset_name}_{self.current_episode_id}"
            )
            for feature_name in self.features.keys():
                if "count" in self.required_stats:
                    additional_metadata[f"{feature_name}_count"] = compacted_table[
                        feature_name
                    ].count()
                if "mean" in self.required_stats:
                    additional_metadata[f"{feature_name}_mean"] = compacted_table[
                        feature_name
                    ].mean()
                if "max" in self.required_stats:
                    additional_metadata[f"{feature_name}_max"] = compacted_table[
                        feature_name
                    ].max()
                if "min" in self.required_stats:
                    additional_metadata[f"{feature_name}_min"] = compacted_table[
                        feature_name
                    ].min()
            self.step_data_connector.save_table(
                f"{self.dataset_name}_{self.current_episode_id}",
            )
            # TODO: this is a hack clear the old dataframe and load as a lazy frame  
            # TODO: future iteration: serve as cache
            self.step_data_connector.load_tables(
                [self.current_episode_id],
                [f"{self.dataset_name}_{self.current_episode_id}"],
            )
        else: 
            table_names = [
                self._get_feature_table_name(feature_name)
                for feature_name in self.features.keys()
            ]
            self.step_data_connector.remove_tables(table_names)

        for metadata_key in additional_metadata.keys():
            logger.debug(f"Adding metadata key {metadata_key} to the database")
            self.episode_info_connector.add_column(
                self.dataset_name,
                metadata_key,
                "str",  # TODO: support more types
            )
        # update the metadata field marking the episode as compacted
        self.episode_info_connector.update_data(
            self.dataset_name, self.current_episode_id, additional_metadata
        )

        self.episode_info_connector.save_table(
            f"{self.dataset_name}"
        )
        self.current_episode_id += 1
