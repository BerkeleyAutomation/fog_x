import time
from typing import Any, Dict, List, Optional

from fog_rtx.feature import FeatureType


class Episode:
    def __init__(
        self,
        description: Optional[str] = None,
        features: Dict[str, FeatureType] = {},
        enable_FeatureType_inferrence=True,  # whether additional FeatureTypes can be inferred
    ):
        self.description = description
        self.features = features
        self.enable_FeatureType_inferrence = enable_FeatureType_inferrence


    def add(
        self,
        feature: str,
        value: Any,
        timestamp: Optional[int] = None,
    ) -> None:
        if timestamp is None:
            timestamp = time.time_ns()

    def close(self) -> None:
        pass
