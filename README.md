# retrieval_ann
Image similarity based annotator with metadata.

# Pipeline:
```
formatted_data: json with the following format
    [
        {
            "url": "...",  # url of image 
            # other metadata fields
        },
        ...
    ]
```
### ---> embedding_model --->
```
embedded_data: chromadb, where each element has
    id - uuid
    embedding of the image
    metadata - all its metadata from fromatted_data
```

### ---> annotation --->
```
annotated_data: json of ids with following format
    [
        {
            "pos": ["id1", "id2", "id3"],
            "neg": ["id4"]  # hard negatives (were shown but not annotated as positives)
        }

        {
            "pos": ["id4", "id5", "id6"],
            "neg": ["id1", "id3"]
        },
        ...
    ]
```


# Annotation through saving to dirs and deleting extra images
### ann_dirrify.py
```
This script creates directory by path ANN_DIRRIFY_DIR_PATH (constant in config). Using sampler specified in configs it generates ANN_DIRRIFY_NDIRS directories. Each directory contains ANN_DIRRIFY_NNEAREST + 1 images (if all images were loaded successfully), image with index 0 is main image and other images are top similar. Directories are named like "{dir index}_{main img id}", and images inside named like "{img index}_{img id}".
Script generates samples.json in ANN_DIRRIFY_DIR_PATH in the following format: 
    {
        "main img id": [
            "item0 id",
            "item1 id", 
            ...
        ],
        ...
    }
This json is needed to detect deleted images, so to make it negatives.

How to properly annotate?
    1) Delete all imgs not similar to main img 
    2) If you don't want results to be saved for some directory, delete this directory.
```

### ann_dirload.py
```
This script loads information from annotated dirs to ANNOTATED_DATA_PATH, if it already exists just adds new information there. 

It goes through directories. 
    Checks if it is in samples.json dict
    Checks if it is already in annotated (if yes it causes message in logs)
    goes through ids in samples[dir_id] 
        1) Checks if id in directory saves as pos else neg

If all these steps performed succesfully, you can delete dir ANN_DIRRIFY_DIR_PATH, to generate new directories.
```
