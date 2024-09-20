import httpx
import asyncio
from typing import List, Tuple, Union, Dict, Callable
from PIL import Image
from io import BytesIO

from .embedders import *

name2model_class: Dict[str, Callable[..., BaseImgEmbedder]] = {
    "clip": CLIPEmbedder
}


class NamedEmbedder:
    def __init__(self, name: str, cfg: str):
        """
            Initialize embedder by its name and config.
        """
        if name not in name2model_class:
            raise Exception(f"No model named {name}")
        self.model = name2model_class[name](**cfg)

    @staticmethod
    async def fetch_image(url: str) -> Union[Image.Image, Exception]:
        """
        Asynchronously fetches an image from a URL. Returns an Image object or an exception.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return e

    async def fetch_images(self, urls: List[str]) -> List[Union[Image.Image, Exception]]:
        """
        Asynchronously fetches a batch of images from URLs. Returns a mix of Image objects and exceptions.
        """
        tasks = [self.fetch_image(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def __call__(self, urls: List[str]) -> Tuple[
            Dict[str, List[float]], 
            Dict[str, Exception]
        ]:
        """
        Asynchronously fetches and processes a batch of image URLs, embedding them and returns two dictionaries:
            Of succesfully loaded urls and embeddings
            Of not succesfully loaded urls and exceptions
        """
        results = await self.fetch_images(urls)

        loaded_urls, loaded_imgs, err_urls, errors = [], [], [], []
        for url, res in zip(urls, results):
            if isinstance(res, Image.Image):
                loaded_urls.append(url)
                loaded_imgs.append(res)
            else:
                err_urls.append(url)
                errors.append(url)
        
        url2emb = {}
        url2err = {}
        if len(loaded_imgs) > 0:
            embs = self.model(loaded_imgs)
            url2emb = {
                url: emb for url, emb in zip(loaded_urls, embs)
            }
        
        if len(err_urls) > 0:
            url2err = {
                url: err for url, err in zip(err_urls, errors)
            }

        return url2emb, url2err
    
    