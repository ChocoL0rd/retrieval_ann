import os
import json
from datetime import datetime
import logging
import asyncio

from tqdm import tqdm
import chromadb
from torch.utils.data import DataLoader

import config
from src.embedders_package import NamedEmbedder
from src.embeddify_utils import deduplicate_data, CustomCollate

log_dir = os.path.join("logs", "annotate")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info(f"FORMATTED_DATA_NAME: {config.FORMATTED_DATA_NAME}")
logger.info(f"COLLECTION_NAME: {config.COLLECTION_NAME}")
logger.info(f"COLLECTION_METRIC: {config.COLLECTION_METRIC}")

async def main():
    # load data
    db = chromadb.PersistentClient(
        path=config.CHROMADB_PATH
    )
    if config.COLLECTION_NAME not in db.list_collections():
        logger.info(f"Collection {config.COLLECTION_NAME} does not exist. Creating ...")
    collection = db.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={
            "hnsw:space": config.COLLECTION_METRIC
        }
    )
    logger.info(f"Number of elements in collection: {collection.count()}")


    with open(config.FORMATTED_DATA_PATH) as f:
        data = json.load(f)
    logger.info(f"Number of elements in formatted data: {len(data)}")

    data = deduplicate_data(data, logger)

    collection_urls = [meta["url"] for meta in collection.get(include=["metadatas"])["metadatas"]]
    data = [item for item in data if item["url"] not in collection_urls]
    logger.info(f"Number of elements remain to be added: {len(data)}")

    loader = DataLoader(
        dataset=data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS,
        collate_fn=CustomCollate()
    )

    # load model
    embedder = NamedEmbedder(
        name=config.MODEL_NAME, 
        cfg=config.MODEL_CFG
    )


    for ids, urls, metas in tqdm(loader):
        url2emb, url2err = await embedder(urls)
        # :TODO : fix for the case when some of the dicts is empty
        embs, ids_of_embs, metas_of_embs = zip(
            *[(url2emb[url], id, meta) for id, url, meta in zip(ids, urls, metas) if url in url2emb]
        )
        collection.add(
            ids=ids_of_embs,
            embeddings=embs,
            metadatas=metas_of_embs
        )

        for url, error in url2err.items():
            logger.error(f"Error for URL {url}: {error}")


    logger.info(f"Number of elements in collection after embedding: {collection.count()}")


if __name__ == "__main__":
    asyncio.run(main())
