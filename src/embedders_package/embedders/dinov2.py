from typing import List
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

from .base import BaseImgEmbedder


class DinoV2Embedder(BaseImgEmbedder):
    def __init__(self, dino_version, device):
        self.processor = AutoImageProcessor.from_pretrained(dino_version)
        self.model = AutoModel.from_pretrained(dino_version).to(device)
        self.model.eval()
        self.device = device


    @torch.no_grad()
    def __call__(self, images: List[Image.Image]) -> List[List[float]]:
        processed = self.processor(images=images, return_tensors="pt")
        outputs = self.model(pixel_values=processed["pixel_values"].to(self.device))
        return outputs.pooler_output.tolist()

