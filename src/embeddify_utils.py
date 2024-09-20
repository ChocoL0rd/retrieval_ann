from typing import List, Tuple
from collections import defaultdict
import uuid


class CustomCollate:
    def __init__(self):
        pass        

    def __call__(self, batch: List[dict]) -> Tuple[List[str], List[str], List[dict]]:
        """ Returns list of ids(uuid), urls and list of metadata excluding those that are not in collection """
        return [str(uuid.uuid4()) for _ in batch], [item["url"] for item in batch], batch


def deduplicate_data(metadatas: List[dict], logger) -> List[dict]:
    """
    Checks for duplicate URLs in the metadata list.
    If such URLs exist, logs them. If no duplicates are found, logs a message indicating that.
    Returns a list of metadata with duplicates removed.
    """
    url_count = defaultdict(int)  # Using defaultdict to count occurrences of each URL
    duplicate_urls = set()        # Set to store duplicate URLs

    # Iterate over the metadata list and count occurrences of each URL
    for meta in metadatas:
        url = meta.get('url')
        if url:
            url_count[url] += 1
            if url_count[url] > 1:
                duplicate_urls.add(url)

    if duplicate_urls:
        logger.warning(f"Duplicate URLs found: {duplicate_urls}")
        logger.info(f"Continuing without all duplicate elements")

        # Filter out elements that have URLs in the duplicate_urls set
        metadatas = [meta for meta in metadatas if meta.get('url') not in duplicate_urls]
        logger.info(f"Number of deduplicated data: {len(metadatas)}")
    else:
        logger.info("No duplicate URLs found.")
    
    return metadatas
