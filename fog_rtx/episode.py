from typing import List, Optional


class Episode:
    def __init__(
        self,
        description: Optional[str] = None,
    ):
        self.description = description
        self.features = []

    def add(self, feature: str, value: str) -> None:
        pass

    def close(self) -> None:
        pass
