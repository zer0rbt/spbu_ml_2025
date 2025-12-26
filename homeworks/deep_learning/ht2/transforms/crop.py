from .base import BaseTransform
from PIL import Image
import random

class RandomCrop(BaseTransform):
    def __init__(self, p: float, crop_size: tuple[int, int], padded: bool = False, mode: str = "RGB"):
        super().__init__(p)
        self.crop_size = crop_size
        self.padded = padded
        self.mode = mode

    def _crop(self, img: Image.Image) -> Image.Image:
        cw, ch = self.crop_size
        w, h = img.size
        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        cropped = img.crop((left, top, left + cw, top + ch))
        if self.padded:
            return cropped.convert(self.mode)
        return cropped.resize((w, h), Image.BILINEAR).convert(self.mode)

    def __call__(self, img: Image.Image) -> Image.Image:
        cw, ch = self.crop_size
        if random.random() > self.p or cw > img.width or ch > img.height:
            return img.convert(self.mode)
        return self._crop(img)