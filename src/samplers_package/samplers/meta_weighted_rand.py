from typing import List, Tuple, Dict
import random
import numpy as np
import chromadb
from .base import BaseAnnSampler, URLMetaPair


class AnnMetaWeightedRandomSampler(BaseAnnSampler):
    def __init__(self, collection: chromadb.Collection, cfg: dict):
        self.collection = collection
        self.cfg = cfg
        
        self.meta_fields = cfg["meta_fields"]
        if isinstance(self.meta_fields, str):
            self.meta_fields = [self.meta_fields]

        # Mappers for each metadata field
        self.mappers: Dict[str, Dict] = {field: {} for field in self.meta_fields}
        
        # Retrieve all ids and metadata
        self.ids = np.array(collection.get()["ids"])
        self.meta_values = np.array([self.map_meta_fields(meta) for meta in collection.get(include=["metadatas"])["metadatas"]])

        # Count the number of occurrences of each metadata category
        self.count_meta_values()

    def map_meta_fields(self, meta: dict) -> Tuple[int, ...]:
        """
        Convert the values of the specified meta_fields into numerical representations (int).
        Use mappers to maintain unique mappings.
        """
        mapped_values = []
        for field in self.meta_fields:
            value = meta[field]
            # If the value hasn't been encoded yet, add it to the mapper
            if value not in self.mappers[field]:
                self.mappers[field][value] = len(self.mappers[field])
            mapped_values.append(self.mappers[field][value])
        return tuple(mapped_values)

    def count_meta_values(self):
        """
        Count the unique combinations of metadata (as tuples) and their occurrence with NumPy.
        All data is already in numerical form (int).
        """
        self.unique_meta_values, self.meta_values_counts = np.unique(self.meta_values, return_counts=True, axis=0)

        # Create probabilities for each category (1 / count) and normalize them
        # self.meta_probabilities = 1.0 / self.meta_values_counts
        # self.meta_probabilities /= self.meta_probabilities.sum()

    def exclude_ids(self, ids: List[str]):
        """
        Exclude the specified ids and recalculate metadata and distribution.
        """
        exclude_ids_set = set(ids)
        mask = np.isin(self.ids, list(exclude_ids_set), invert=True)

        # Update ids and meta_values, filtering out the excluded ids
        self.ids = self.ids[mask]
        self.meta_values = self.meta_values[mask]

        # Recalculate metadata and distribution after removing ids
        self.count_meta_values()

    def __call__(self, n_nearest: int) -> Tuple[URLMetaPair, List[URLMetaPair]]:
        if len(self.ids) == 0:
            return None

        # Step 1: Randomly choose a category based on probabilities
        chosen_meta_value = self.unique_meta_values[np.random.choice(
            len(self.unique_meta_values), 
            # p=self.meta_probabilities
        )]

        # Step 2: Find all ids in the selected category (using NumPy to filter)
        mask = np.all(self.meta_values == chosen_meta_value, axis=1)
        available_ids = self.ids[mask]

        # Step 3: Randomly select an element from this category
        main_id = np.random.choice(available_ids)
        main_item_data = self.collection.get(
            ids=main_id.tolist(),  # Convert back to standard list for the query
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

        # Step 4: Find the nearest elements
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
