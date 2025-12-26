from .base_layer import BaseLayer
import numpy as np
class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (self.out * (1.0 - self.out))