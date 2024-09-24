import os
import json
from datetime import datetime
import logging
from tqdm import tqdm
import chromadb
import aiohttp
import asyncio

import config
from src.samplers_package import NamedSampler

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
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

logger.info(f"COLLECTION_NAME: {config.COLLECTION_NAME}")
logger.info(f"CHROMADB_PATH: {config.CHROMADB_PATH}")
logger.info(f"ANNOTATED_DATA_PATH: {config.ANNOTATED_DATA_PATH}")
logger.info(f"ANN_SAMPLER_NAME: {config.ANN_SAMPLER_NAME}")
logger.info(f"ANN_SAMPLER_CFG: {config.ANN_SAMPLER_CFG}")
logger.info(f"ANN_DIRRIFY_DIR_PATH: {config.ANN_DIRRIFY_DIR_PATH}")
logger.info(f"ANN_DIRRIFY_NDIRS: {config.ANN_DIRRIFY_NDIRS}")
logger.info(f"ANN_DIRRIFY_NNEAREST: {config.ANN_DIRRIFY_NNEAREST}")

os.makedirs(config.ANN_DIRRIFY_DIR_PATH, exist_ok=False)

# read annotated data if exists
annotated_ids = set()
if os.path.exists(config.ANNOTATED_DATA_PATH):
    with open(config.ANNOTATED_DATA_PATH) as f:
        annotated_data = json.load(f)
        annotated_ids = {pos for ann in annotated_data for pos in ann["pos"]}
else:
    with open(config.ANNOTATED_DATA_PATH, "w") as f:
        json.dump([], f, indent=4)

logger.info(f"Number of annotated ids: {len(annotated_ids)}")

# read collection
db = chromadb.PersistentClient(
    path=config.CHROMADB_PATH
)

collection = db.get_collection(
    name=config.COLLECTION_NAME,
)
logger.info(f"Number of elements in collection: {collection.count()}")

sampler = NamedSampler(
    collection=collection, 
    meta_fields=config.ANN_META_FIELDS,
    name=config.ANN_SAMPLER_NAME,
    cfg=config.ANN_SAMPLER_CFG
)
sampler.exclude_ids(annotated_ids)

samples = {}


async def download_image(session, item, dir_path, i):
    try:
        async with session.get(item.url) as resp:
            if resp.status == 200:
                image_path = os.path.join(dir_path, f"{i}_{item.id}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(await resp.read())
                return item.id
            else:
                logger.info(f"Error loading image {item.id}, URL: {item.url}, status code: {resp.status}")
                return None
    except Exception as e:
        logger.info(f"Error {e} loading image: {item.id}, URL: {item.url}")
        return None


async def process_dir(n_dir, main_item, nearest_items, dir_name, dir_path):
    async with aiohttp.ClientSession() as session:
        successfully_loaded_items = []
        for i, item in enumerate([main_item] + nearest_items):
            item_id = await download_image(session, item, dir_path, i)
            if item_id:
                successfully_loaded_items.append(item_id)
            elif i == 0:
                # If main_item failed to load, skip the entire directory and its items
                logger.info(f"Main item {main_item.id} failed to load, skipping directory {dir_name}")
                os.rmdir(dir_path)
                return None
        # If all successfully loaded
        samples[main_item.id] = successfully_loaded_items
        return successfully_loaded_items


async def main():
    for n_dir in tqdm(range(config.ANN_DIRRIFY_NDIRS)):
        sample_res = sampler(config.ANN_DIRRIFY_NNEAREST)
        if sample_res is None:
            logger.info("No items remaining")
            break

        main_item, nearest_items = sample_res
        dir_name = f"{n_dir}_{main_item.id}"
        dir_path = os.path.join(config.ANN_DIRRIFY_DIR_PATH, dir_name)
        os.mkdir(dir_path)

        # Process directory and load images
        successfully_loaded_items = await process_dir(n_dir, main_item, nearest_items, dir_name, dir_path)

        if successfully_loaded_items:
            # Exclude successfully loaded items from the sampler
            sampler.exclude_ids(successfully_loaded_items)


if __name__ == "__main__":
    asyncio.run(main())

# After the process, save annotated ids and samples to disk
with open(os.path.join(config.ANN_DIRRIFY_DIR_PATH, "samples.json"), "w") as f:
    json.dump(samples, f, indent=4)

logger.info(f"Finished processing, saved {len(samples)} directories")
