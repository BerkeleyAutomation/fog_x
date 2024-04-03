import time
from typing import Any, Dict, List, Optional

from fog_rtx.database import DatabaseManager
from fog_rtx.feature import FeatureType

import logging
logger = logging.getLogger(__name__)

class Episode:
    def __init__(
        self,
        description: Optional[str] = None,
        features: Dict[str, FeatureType] = {},
        enable_feature_inferrence=True,  # whether additional FeatureTypes can be inferred
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.description = description
        self.features = features
        self.enable_feature_inferrence = enable_feature_inferrence
        self.db_manager = db_manager

    def add(
        self,
        feature: str,
        value: Any,
        timestamp: Optional[int] = None,
    ) -> None:
        if timestamp is None:
            timestamp = time.time_ns()
        
        logger.info(f"Adding {feature} with value {value} at timestamp {timestamp}")
        if self.db_manager:
            self.db_manager.add(feature, value)
        
        


    def close(self) -> None:
        pass
