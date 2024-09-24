import os
import json
from datetime import datetime
import logging
import chromadb

import config

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

logger.info(f"ANNOTATED_DATA_PATH: {config.ANNOTATED_DATA_PATH}")
logger.info(f"ANN_DIRRIFY_DIR_PATH: {config.ANN_DIRRIFY_DIR_PATH}")

# read annotated data if exists
annotated_ids = set()
if os.path.exists(config.ANNOTATED_DATA_PATH):
    with open(config.ANNOTATED_DATA_PATH) as f:
        annotated_data = json.load(f)
        annotated_ids = {pos for ann in annotated_data for pos in ann["pos"]}
else:
    with open(config.ANNOTATED_DATA_PATH, "w") as f:
        json.dump([], f, indent=4)

logger.info(f"Number of annotated ids before dirload: {len(annotated_ids)}")


samples_json_path = os.path.join(config.ANN_DIRRIFY_DIR_PATH, "samples.json")
with open(samples_json_path) as f:
    samples_info = json.load(f)


new_annotated = []
for entry in os.scandir(config.ANN_DIRRIFY_DIR_PATH):
    pos = []
    neg = []

    dir_path = entry.path
    dir_name = entry.name.split("_")[-1]

    if dir_name in annotated_ids:
        logger.warning(f"Meet dir with path {dir_path}, but id {dir_name} already in annotated, skip ...")
        continue

    if not entry.is_dir():
        logger.warning(f"Path {dir_path} is not directory, skip ...")
        continue

    listdir = os.listdir(dir_path)
    item_names = [
        name.split(".")[0].split("_")[-1] 
        for name in listdir
    ]
    for sample_item_name in samples_info[dir_name]:
        if sample_item_name in item_names:
            pos.append(sample_item_name)
        else:
            neg.append(sample_item_name)

    # little extra check
    for item_name, file_name in zip(item_names, listdir):
        if item_name not in samples_info[dir_name]:
            logging.warning(f"Name {item_name} of file {file_name} in directory {dir_name} is not in samples.json by key {dir_name}")

    new_annotated.append({
        "pos": pos,
        "neg": neg
    })


annotated_data += new_annotated

with open(config.ANNOTATED_DATA_PATH, "w") as f:
    json.dump(
        annotated_data,
        f,
        indent=4
    )

annotated_ids = {pos for ann in annotated_data for pos in ann["pos"]}
logger.info(f"Number of annotated ids after dirload: {len(annotated_ids)}")



