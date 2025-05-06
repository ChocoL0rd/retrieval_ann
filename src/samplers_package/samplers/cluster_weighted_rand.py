from typing import List, Tuple
import random
import chromadb
from .base import BaseAnnSampler, URLMetaPair
from sklearn.cluster import KMeans


class AnnClusterWeightedRandomSampler(BaseAnnSampler):
    def __init__(self, collection: chromadb.Collection, cfg: dict):
        self.collection = collection
        self.ids: List[str] = collection.get()["ids"]
        self.cfg = cfg
        self.cluster = KMeans(
            n_clusters=cfg["n_clusters"],
            max_iter=cfg["max_iter"],
            random_state=cfg["random_state"]
        )
        if isinstance(cfg["train_size"], int):
            if cfg["train_size"] > len(self.ids):
                raise ValueError(f"train_size has to be less then collection size {len(self.ids)} not: {cfg['train_size']}")
            train_size = cfg["train_size"]
        elif isinstance(cfg["train_size"], float):
            if not (0 < cfg["train_size"] <= 1):
                raise ValueError(f"train_size has to be in (0, 1] not: {cfg['train_size']}")
            train_size = int(cfg["train_size"] * len(self.ids))
        else:
            raise TypeError(f"train_size has to be int of float not: {type(cfg['train_size'])}")
        
        random.seed(cfg["random_state"])
        train_ids = random.sample(self.ids, k=train_size)
        train_data = self.collection.get(train_ids, include=["embeddings"])["embeddings"]
        self.cluster.fit(train_data)
        self.buffer_size = cfg["buffer_size"]
        

    def create_buffer(self):
        self.buffer = random.sample(self.ids, k=self.buffer_size)


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
