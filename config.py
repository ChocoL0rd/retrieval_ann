import os

# model configs
MODEL_NAME = "clip"
MODEL_CFG = {
    "device": "cuda",
    "clip_version": "patrickjohncyh/fashion-clip"
}
BATCH_SIZE = 256
PIN_MEMORY = True
NUM_WORKERS = 4


# data configs
FORMATTED_DATA_NAME = "tmp"
COLLECTION_NAME = "tmp"
ANNOTATED_DATA_NAME = "tmp"

COLLECTION_METRIC = "cosine"  # l2, ip (inner product), cosine

# annotation configs
ANN_META_FIELDS = ["url"]


# default paths
FORMATTED_DIR = "data/formatted"
EMBEDDED_DIR = "data/embedded"
ANNOTATED_DIR = "data/annotated"
CHROMADB_NAME = "chroma"


FORMATTED_DATA_PATH = os.path.join(FORMATTED_DIR, f"{FORMATTED_DATA_NAME}.json")
CHROMADB_PATH = os.path.join(EMBEDDED_DIR, f"{CHROMADB_NAME}")
ANNOTATED_DATA_PATH = os.path.join(ANNOTATED_DIR, f"{ANNOTATED_DATA_NAME}.json")



