from typing import Any, List, Optional
import time

from fog_rtx.feature import FeatureType 

class Episode:
    def __init__(
        self,
        description: Optional[str] = None,
        feature: List[FeatureType] = [],
        enable_FeatureType_inferrence = True,  # whether additional FeatureTypes can be inferred
    ):
        self.description = description
        self.feature = []

    def add(self, 
            feature: str, 
            value: Any, 
            timestamp: Optional[int] = None,
            ) -> None:
        if timestamp is None:
            timestamp = time.time()


    def close(self) -> None:
        pass
