import logging
from typing import Dict, List, Optional, Tuple

from fog_rtx.episode import Episode
from fog_rtx.feature import FeatureType
from fog_rtx.db import Database

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
        features: Dict[str, FeatureType] = {},  # features to be stored {name: FeatureType}
        enable_feature_inferrence=True,  # whether additional features can be inferred
        database: Optional[Database] = None,
    ) -> None:
        self.name = name
        self.path = path
        self.replace_existing = replace_existing
        self.features = features
        self.enable_feature_inferrence = enable_feature_inferrence

        logger.warn(
            f"Dataset {name} created at {path} with features {features}"
        )

    def new_episode(self, description: str) -> Episode:
        """
        Create a new episode / trajectory.
        """
        return Episode(description)

    def query(self, query: str) -> List[Episode]:
        """
        Query the dataset.
        """
        return []

    def export(self, path: str, format: str) -> None:
        """
        Export the dataset.
        """
        if format == "rtx":
            pass
        else:
            raise ValueError("Unsupported export format")
