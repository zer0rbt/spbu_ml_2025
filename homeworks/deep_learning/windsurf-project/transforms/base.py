from PIL import Image

class BaseTransform:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        raise NotImplementedError("U CALLED THE PLACEHOLDER")