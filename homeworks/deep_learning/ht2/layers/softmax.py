from .base_layer import BaseLayer
import numpy as np
class Softmax(BaseLayer):
    def __init__(self): super().__init__(); self.probs = None
    def forward(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.probs

    def backward(self, dout):
        return self.probs * (dout - np.sum(self.probs * dout, axis=1, keepdims=True))