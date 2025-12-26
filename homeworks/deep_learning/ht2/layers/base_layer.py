import numpy as np

class BaseLayer:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        """Возвращает список кортежей (param, grad)"""
        return []

    def zero_grad(self):
        for _, grad in self.parameters():
            if grad is not None:
                grad.fill(0)

    def __call__(self, x):
        return self.forward(x)