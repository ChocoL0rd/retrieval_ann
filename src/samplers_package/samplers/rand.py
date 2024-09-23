from typing import List, Tuple
import random
import chromadb
from .base import BaseAnnSampler, URLMetaPair


class AnnRandomSampler(BaseAnnSampler):
    def __init__(self, collection: chromadb.Collection, cfg: dict):
        self.collection = collection
        self.ids: List[str] = collection.get()["ids"]


    def exclude_ids(self, ids: List[str]):
        self.ids = [id for id in self.ids if id not in ids]


    def __call__(self, n_nearest: int) -> Tuple[URLMetaPair, List[URLMetaPair]]:
        if len(self.ids) == 0:
            return None
        i = random.randint(0, len(self.ids) - 1)
        main_id = self.ids[i]
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
