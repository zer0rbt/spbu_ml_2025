from .base import BaseTransform
from PIL import Image
import random

class RandomZoom(BaseTransform):
    def __init__(self, p: float, zoom_range: tuple[float, float], padded: bool = False, mode: str = "RGB"):
        super().__init__(p)
        self.zoom_range = zoom_range
        self.padded = padded
        self.mode = mode

    def _zoom(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        zoom_factor = random.uniform(*self.zoom_range)
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        zoomed = img.resize((new_w, new_h), Image.BILINEAR)
        if not self.padded:
            return zoomed.resize((w, h), Image.BILINEAR).convert(self.mode)
        result = Image.new(self.mode, (w, h))
        left = max((w - new_w) // 2, 0)
        top = max((h - new_h) // 2, 0)
        result.paste(zoomed, (left, top))
        return result.convert(self.mode)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img.convert(self.mode)
        return self._zoom(img)
