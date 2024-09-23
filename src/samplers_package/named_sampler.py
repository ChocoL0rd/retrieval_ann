from typing import Optional, Tuple, List
import chromadb

from .samplers import *
import logging

name2sampler_class = {
    "rand": AnnRandomSampler,
    "meta_weighted_rand": AnnMetaWeightedRandomSampler
}


class NamedSampler:
    def __init__(self, collection: chromadb.Collection,  meta_fields: dict, name: str, cfg: dict):
        self.sampler: BaseAnnSampler = name2sampler_class[name](collection, cfg)
        self.meta_fields = meta_fields


    def exclude_ids(self, ids: List[str]) -> None:
        self.sampler.exclude_ids(ids)


    def __call__(self, n_nearest: int) -> Optional[Tuple[URLMetaPair, List[URLMetaPair]]]:
        sample_res = self.sampler(n_nearest)
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
    
