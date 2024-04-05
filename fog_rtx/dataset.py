import logging
from typing import Dict, List, Optional, Tuple, Any

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
            db_connector = DatabaseConnector(f"{path}/rtx.db")
        self.db_manager = DatabaseManager(db_connector)
        self.db_manager.initialize_dataset(self.name, features)

        self.storage = storage

    def new_episode(self, 
                    metadata: Optional[Dict[str, Any]] = None) -> Episode:
        """
        Create a new episode / trajectory.
        TODO #1: support multiple processes writing to the same episode
        TODO #2: close the previous episode if not closed
        """
        return Episode(
            metadata=metadata,
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
        return self.db_manager.get_metadata_table("sql")

    def get_metadata_as_pandas_df(self):
        """
        Return the metadata as pandas dataframe.
        """
        return self.db_manager.get_metadata_table("pandas")

    def load(
        self,
        dataset_path: str,
        format: str,
        split: Optional[str],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Load the dataset.
        """
        if format != "rtx":
            raise ValueError("Unsupported format")

        # this is only required if rtx format is used
        import tensorflow_datasets as tfds

        from fog_rtx.rlds.utils import dataset2path

        b = tfds.builder_from_directory(builder_dir=dataset2path(dataset_path))
        ds = b.as_dataset(split=split)
        for tf_episode in ds:
            logger.info(tf_episode)
            fog_epsiode = self.new_episode(
                metadata = additional_metadata, 
            )
            for step in tf_episode["steps"]:
                for k, v in step.items():
                    if k == "observation" or k == "action":
                        for k2, v2 in v.items():
                            fog_epsiode.add(
                                feature=str(k2),
                                value=str(v2.numpy()),
                                feature_type=FeatureType(dtype="float64"),
                            )
                    else:
                        fog_epsiode.add(
                            feature=str(k),
                            value=str(v.numpy()),
                            feature_type=FeatureType(dtype="float64"),
                        )
            fog_epsiode.close()

    def read_by(self, pandas_metadata: Any = None):
        episode_ids = list(pandas_metadata["id"])
        logger.info(f"Reading episodes as order: {episode_ids}")
        episodes = []
        for episode_id in episode_ids:
            episodes.append(self.db_manager.get_episode_table(episode_id))
        return episodes