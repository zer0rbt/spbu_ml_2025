import torch


class Node:
    def __init__(self, data, _children=(), _op=''):
        self.data = torch.tensor(data, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        self.grad = torch.tensor(0.0, dtype=torch.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Node(data={self.data.item() if self.data.numel() == 1 else self.data.tolist()}, grad={self.grad.item() if self.grad.numel() == 1 else self.grad.tolist()})"
    
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), '+')
        
        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), '*')
        
        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def relu(self):
        out = Node(torch.clamp(self.data, min=0), (self,), 'relu')
        
        def _backward():
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (out.data > 0).float() * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = torch.tensor(1.0, dtype=torch.float32)
        
        for node in reversed(topo):
            node._backward()
