from inspect import signature

import torch
from torch import nn


class SymbolicActivation(nn.Module):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = [(f.name, f.torch) for f in funcs]

    def forward(self, x):
        out = []
        in_i = 0
        for name, func in self.funcs:
            if len(signature(func).parameters) == 1:
                out.append(func(x[:, in_i]))
                in_i += 1
            else:
                out.append(func(x[:, in_i], x[:, in_i + 1]))
                in_i += 2
        return torch.stack(out, dim=1)
