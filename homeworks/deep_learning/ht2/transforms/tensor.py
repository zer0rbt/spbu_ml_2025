import torch
from PIL import Image

class ToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        h, w = img.size[1], img.size[0]
        c = len(img.getbands())
        return data.reshape(h, w, c).permute(2, 0, 1).float() / 255.0
