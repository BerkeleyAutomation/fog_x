
from typing import List, Tuple
from .episode import Episode

class Dataset:
    """
    Create or load from a new dataset.
    """
    def __init__(self,
                    name : str,
                    path : str,
                    replace_existing : bool = False,
                    ) -> None:
        self.name = name
        self.path = path
        self.replace_existing = replace_existing

    def new_episode(self, description : str) -> Episode:
        """
        Create a new episode / trajectory.
        """
        return Episode(description)
    
    def query(self, query : str) -> List[Episode]:
        """
        Query the dataset.
        """
        return []
    
    def export(self, path : str, format : str) -> None:
        """
        Export the dataset.
        """
        if format == "rtx":
            pass
        else:
            raise ValueError("Unsupported export format")