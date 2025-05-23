from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class BaseImgEmbedder(ABC):
    @abstractmethod
    def __call__(self, imgs: List[Image.Image]) -> List[List[float]]:
        pass