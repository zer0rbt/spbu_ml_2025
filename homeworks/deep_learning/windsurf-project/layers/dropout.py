from .base_layer import BaseLayer
import numpy as np
class Dropout(BaseLayer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if self.training and self.p > 0:
            self.mask = (np.random.rand(*x.shape) >= self.p)
            return x * self.mask / (1.0 - self.p)
        else:
            self.mask = None
            return x

    def backward(self, dout):
        if not self.training:
            raise RuntimeError("BACKWARD BROKE")
        return dout * self.mask / (1.0 - self.p)