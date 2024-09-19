from typing import List
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from .base import BaseImgEmbedder


class CLIPEmbedder(BaseImgEmbedder):
    def __init__(self, clip_version, device):
        self.model = CLIPModel.from_pretrained(clip_version).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(clip_version)
        self.device = device


    @torch.no_grad()
    def __call__(self, images: List[Image.Image]) -> List[List[float]]:
        processed = self.processor(images=images, return_tensors="pt")
        embs = self.model.get_image_features(pixel_values=processed["pixel_values"].to(self.device))
        return embs.tolist()

