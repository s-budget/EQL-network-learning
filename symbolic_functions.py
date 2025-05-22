import numpy as np
import torch


class Const1:
    name = "1"

    def torch(self, x):
        return torch.ones_like(x)/1


class Identity:
    name = "x"

    def torch(self, x):
        return x/1


class Square:
    name = "(x)^2"

    def torch(self, x):
        return torch.square(x)/1


class Sin2Pi:
    name = "sin(2Pi*(x))"

    def torch(self, x):
        return torch.sin(x * 2  * np.pi)/1


class Exp:
    name = "(e^(x)-1)/e"

    def torch(self, x):
        return (torch.exp(x) - 1) / np.e


class SigmoidScaled:
    name = "sigmoid(x)"

    def torch(self, x):
        return torch.sigmoid(x)/1


class Multiply:
    name = "(x1) * (x2)"

    def torch(self, x1, x2):
        return x1 * x2/1


def count_double(funcs):
    from inspect import signature
    return sum(1 for f in funcs if len(signature(f.torch).parameters) == 2)
