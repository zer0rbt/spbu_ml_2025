from .base_layer import BaseLayer
import numpy as np
class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
