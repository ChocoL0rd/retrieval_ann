import streamlit as st

import os
import json
from datetime import datetime
import logging

import chromadb

import config
from src.annotator_utils import AnnRandomSampler, URLMetaPair


if "initialized" not in st.session_state:
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
    logger.info(f"ANNOTATED_DATA_NAME: {config.ANNOTATED_DATA_NAME}")


    # read annotated data if exists
    annotated_ids = set()
    if os.path.exists(config.ANNOTATED_DATA_PATH):
        with open(config.ANNOTATED_DATA_PATH) as f:
            annotated_data = json.load(f)
            annotated_ids = {pos for ann in annotated_data for pos in ann["pos"]}
    else:
        with open(config.ANNOTATED_DATA_PATH, "w") as f:
            json.dump(
                [],
                f,
                indent=4
            )
        

    logger.info(f"Number of annotated ids: {len(annotated_ids)}")


    # read collection
    db = chromadb.PersistentClient(
        path=config.CHROMADB_PATH
    )

    collection = db.get_collection(
        name=config.COLLECTION_NAME,
    )
    logger.info(f"Number of elements in collection: {collection.count()}")

    sampler = AnnRandomSampler(collection=collection, meta_fields=config.ANN_META_FIELDS)
    sampler.exclude_ids(annotated_ids)

    # Store variables in session_state to persist across runs
    st.session_state["annotated_data"] = annotated_data
    st.session_state["annotated_ids"] = annotated_ids
    st.session_state["collection"] = collection
    st.session_state["sampler"] = AnnRandomSampler(collection=collection, meta_fields=config.ANN_META_FIELDS)
    st.session_state["sampler"].exclude_ids(annotated_ids)
    st.session_state["initialized"] = True


n_nearest = st.slider("Number of nearest elements", 1, 20, 5)

if st.button("Sample"):
    sample_res = st.session_state["sampler"](n_nearest)
    if sample_res is None:
        st.write("No items remaining")
    else:
        main_item, nearest_items = sample_res
        id2pos_neg = {item.id: False for item in nearest_items}

        st.subheader("Main Element")
        st.image(main_item.url, caption=main_item.id)
        st.write(f"Metadata:", main_item.metadata)

        st.subheader(f"Top {n_nearest} Nearest Elements")
        for i, item in enumerate(nearest_items):            
            id2pos_neg[nearest_items[i].id] = st.checkbox(f"Element {i}")
            st.image(item.url, caption=nearest_items[i].id)
            st.write("Metadata:", item.metadata)

        
        if st.button("Save and Proceed"):
            pos = [id for id, value in id2pos_neg.items() if value]
            neg = [id for id, value in id2pos_neg.items() if not value]
            st.session_state["annotated_data"].append(
                {
                    "pos": pos,
                    "neg": neg
                }
            )
            with open(config.ANNOTATED_DATA_PATH, "w") as f:
                json.dump(
                    st.session_state["annotated_data"],
                    f,
                    indent=4
                )
        
            st.session_state["sampler"].exclude_ids(pos)

