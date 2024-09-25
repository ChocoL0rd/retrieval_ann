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
# ANN_SAMPLER_NAME = "rand"
# ANN_SAMPLER_CFG = {}

ANN_SAMPLER_NAME = "meta_weighted_rand"
ANN_SAMPLER_CFG = {
    "meta_fields": "url"
}

ANN_DIRRIFY_DIR_PATH = os.path.join("data", "dirrify", "tmp")
ANN_DIRRIFY_NDIRS = 1000
ANN_DIRRIFY_NNEAREST = 20

# default paths
FORMATTED_DIR = "data/formatted"
EMBEDDED_DIR = "data/embedded"
ANNOTATED_DIR = "data/annotated"
CHROMADB_NAME = "chroma"


FORMATTED_DATA_PATH = os.path.join(FORMATTED_DIR, f"{FORMATTED_DATA_NAME}.json")
CHROMADB_PATH = os.path.join(EMBEDDED_DIR, f"{CHROMADB_NAME}")
ANNOTATED_DATA_PATH = os.path.join(ANNOTATED_DIR, f"{ANNOTATED_DATA_NAME}.json")



