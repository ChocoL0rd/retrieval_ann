from typing import List, Tuple, Optional, Union
import random

from abc import ABC, abstractmethod
import logging

from pydantic import BaseModel

import chromadb


logger = logging.getLogger(__name__)


class URLMetaPair(BaseModel):
    id: str
    url: str
    metadata: dict


class BaseAnnSampler(ABC):
    def __init__(self, meta_fields: Optional[Union[List[str], str]]):
        self.meta_fields = None
        if isinstance(meta_fields, str):
            self.meta_fields = [meta_fields]


    @abstractmethod
    def sample(self, n_nearest: int) -> Optional[Tuple[URLMetaPair, List[URLMetaPair]]]:
        """ Returns some item and list of nearest items, if no remaining returns None"""
        pass 


    def __call__(self, n_nearest: int) -> Optional[Tuple[URLMetaPair, List[URLMetaPair]]]:
        sample_res = self.sample(n_nearest)
        # no samples remaining
        if sample_res is None:
            return None
        
        # there are samples
        main_item, nearest_items = sample_res

        # filter metadata fields
        if self.meta_fields is None:
            return main_item, nearest_items

        # only specified meta fields are remaining
        main_item.metadata = {field: main_item.metadata.get(field) for field in self.meta_fields}
        for item in nearest_items:
            item.metadata = {field: item.metadata.get(field) for field in self.meta_fields}

        return main_item, nearest_items


class AnnRandomSampler(BaseAnnSampler):
    def __init__(self, collection: chromadb.Collection,  meta_fields):
        self.collection = collection
        self.ids: List[str] = collection.get()["ids"]
        
        super().__init__(meta_fields)


    def exclude_ids(self, ids: List[str]):
        self.ids = [id for id in self.ids if id not in ids]


    def sample(self, n_nearest: int) -> Tuple[URLMetaPair, List[URLMetaPair]]:
        if len(self.ids) == 0:
            return None
        i = random.randint(0, len(self.ids) - 1)
        main_id = self.ids.pop(i)
        main_item_data = self.collection.get(
            ids=main_id,
            include=["embeddings", "metadatas"]
        )
        main_emb = main_item_data["embeddings"][0]
        main_meta = main_item_data["metadatas"][0]
        main_url = main_meta["url"]
        main_item = URLMetaPair(
            id=main_id,
            url=main_url,
            metadata=main_meta
        )

        nearest_data = self.collection.query(
            query_embeddings=[main_emb],
            n_results=n_nearest + 1,
            include=["metadatas"]
        )
        nearest_items = [
            URLMetaPair(
                id=id,
                url=meta["url"],
                metadata=meta
            ) 
            for id, meta in zip(nearest_data["ids"][0], nearest_data["metadatas"][0]) if meta["url"] != main_url
        ]
        
        return main_item, nearest_items
