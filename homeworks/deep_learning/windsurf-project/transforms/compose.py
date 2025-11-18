from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: Image.Image):
        for t in self.transforms:
            img = t(img)
        return img
