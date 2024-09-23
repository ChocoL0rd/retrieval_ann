from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel


class URLMetaPair(BaseModel):
    id: str
    url: str
    metadata: dict


class BaseAnnSampler(ABC):
    @abstractmethod
    def __call__(self, n_nearest: int) -> Optional[Tuple[URLMetaPair, List[URLMetaPair]]]:
        """ Returns some item and list of nearest items, if no remaining returns None"""
        pass 


    @abstractmethod
    def exclude_ids(self, ids: List[str]) -> None:
        """ To guarantee that ids are going to be repeated """
        pass


