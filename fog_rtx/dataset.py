import logging
from typing import Dict, List, Optional, Tuple

from fog_rtx.database import DatabaseConnector, DatabaseManager
from fog_rtx.episode import Episode
from fog_rtx.feature import FeatureType

logger = logging.getLogger(__name__)


class Dataset:
    """
    Create or load from a new dataset.
    """

    def __init__(
        self,
        name: str,
        path: str,
        replace_existing: bool = False,
        features: Dict[
            str, FeatureType
        ] = {},  # features to be stored {name: FeatureType}
        enable_feature_inferrence=True,  # whether additional features can be inferred
        db_connector: Optional[DatabaseConnector] = None,
        storage: Optional[str] = None,
    ) -> None:
        self.name = name
        self.path = path
        self.replace_existing = replace_existing
        self.features = features
        self.enable_feature_inferrence = enable_feature_inferrence
        if db_connector is None:
            logger.error(
                "TODO: haven't implemented db_connector generation yet"
            )
            raise NotImplementedError
        self.db_manager = DatabaseManager(db_connector)
        self.db_manager.initialize_dataset(self.name, features)

        self.storage = storage

    def new_episode(self, description: str) -> Episode:
        """
        Create a new episode / trajectory.
        TODO: support multiple processes writing to the same episode
        """
        return Episode(
            description=description,
            features=self.features,
            enable_feature_inferrence=self.enable_feature_inferrence,
            db_manager=self.db_manager,
        )

    def query(self, query: str) -> List[Episode]:
        """
        Query the dataset.
        """
        return self.db_manager.query(query)

    def export(self, path: str, format: str) -> None:
        """
        Export the dataset.
        """
        if format == "rtx":
            pass
        else:
            raise ValueError("Unsupported export format")

    def get_metadata_as_sql(self):
        """
        Return the metadata as SQL.
        """
        return self.db_manager.get_metadata_table(
            "sql"
        )
    
    def get_metadata_as_pandas_df(self):
        """
        Return the metadata as pandas dataframe.
        """
        return self.db_manager.get_metadata_table(
            "pandas"
        )
    