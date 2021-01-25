"""
监控参数梯度的变化
"""
import torch
from inspect import isgenerator

class GradMonitor(object):
    def __init__(self, parameters):
        super().__init__()
        assert isgenerator(parameters) or isinstance(parameters, list)
        self.parameters = parameters

    def __call__(self, ord=1):
        grad_norm = []
        for p in self.parameters:
            if p.requires_grad:
                norm = p.grad.norm(ord)
                grad_norm.append(norm)
            else:
                continue
        grad_norm = torch.tensor(grad_norm, dtype=torch.float)
        return float(grad_norm.norm(ord))

