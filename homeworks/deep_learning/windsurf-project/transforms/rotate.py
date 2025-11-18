from .base import BaseTransform
import random
from PIL import Image

from .base import BaseTransform
import random
from PIL import Image

class RandomRotate(BaseTransform):
    def __init__(self, p: float, degrees: float, mode: str = "RGB"):
        super().__init__(p)
        self.degrees = degrees
        self.mode = mode

    def _rotate(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0).resize((w, h)).convert(self.mode)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img.convert(self.mode)
        return self._rotate(img)