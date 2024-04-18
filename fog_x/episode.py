import logging
import time
from typing import Any, Dict, List, Optional

from fog_x.database import DatabaseManager
from fog_x.feature import FeatureType

logger = logging.getLogger(__name__)


class Episode:
    def __init__(
        self,
        db_manager: DatabaseManager,
        metadata: Optional[Dict[str, Any]] = {},
        features: Dict[str, FeatureType] = {},
        enable_feature_inference=True,  # whether additional FeatureTypes can be inferred
    ):
        self.features = features
        self.enable_feature_inference = enable_feature_inference
        self.db_manager = db_manager
        self.db_manager.initialize_episode(additional_metadata=metadata)

    def add(
        self,
        feature: str,
        value: Any,
        timestamp: Optional[int] = None,
        feature_type: Optional[FeatureType] = None,
        metadata_only = False,
    ) -> None:
        """
        Add one feature step data.
        To add multiple features at each step, call this multiple times or use
        `add_by_dict` to ensure the same timestamp is used for each feature.

        Args:
            feature (str): name of the feature
            value (Any): value associated with the feature
            timestamp (optional int): nanoseconds since the Epoch.
                If not provided, the current time is used.

        Examples:
            >>> episode.add('feature1', 'image1.jpg')
            >>> episode.add('feature2', 100)
        """

        if timestamp is None:
            timestamp = time.time_ns()

        logger.debug(
            f"Adding {feature} with value {value} at timestamp {timestamp}"
        )
        if self.db_manager:
            self.db_manager.add(feature, value, timestamp, feature_type, metadata_only)
        else:
            logger.warning(
                "No database manager provided, data will not be saved"
            )

    def add_by_dict(
        self, data: Dict[str, Any], timestamp: Optional[int] = None
    ) -> None:
        """
        Add multiple features as step data.
        Ensures that the same timestamp is used for each feature.

        Args:
            data (Dict[str, Any]): feature names and their values
            timestamp (optional int): nanoseconds since the Epoch.
                If not provided, the current time is used.

        Examples:
            >>> episode.add_by_dict({'feature1': 'image1.jpg', 'feature2': 100})
        """
        if timestamp is None:
            timestamp = time.time_ns()
        for feature, value in data.items():
            self.add(feature, value, timestamp)

    def compact(self) -> None:
        """
        Creates a table for the compacted data.

        TODO:
            * compact should not be run more than once?
            * expand docstring description
        """
        self.db_manager.compact()

    def get_steps(self) -> List[Dict[str, Any]]:
        """
        Retrieves the episode's step data.

        Returns:
            the step data

        TODO:
            * get_steps not in db_manager; db_manager.get_step_table_all returns a `LazyFrame`, not `List[Dict[str, Any]]`
        """
        return self.db_manager.get_steps()

    def close(self, save_data = True, save_metadata = True, additional_metadata = {}) -> None:
        """
        Saves the episode object.
        """
        self.db_manager.close(save_data=save_data, save_metadata=save_metadata, additional_metadata=additional_metadata)
