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
