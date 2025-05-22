import copy
from inspect import signature

import torch
from torch import nn as nn

from Seminar.symbolic_activation import SymbolicActivation

# Dataset parameters
N_TRAIN = 256  # Number of training examples
DOMAIN = (-2, 2)  # Training domain for input features
N_TEST = 100  # Number of test examples
DOMAIN_TEST = (-2, 2)  # Test domain (for extrapolation)

def generate_data(func, N, range_min=None, range_max=None, noise_std=0.0):
    range_min = range_min if range_min is not None else DOMAIN[0]
    range_max = range_max if range_max is not None else DOMAIN[1]

    input_dim = len(signature(func).parameters)
    x = torch.rand(N, input_dim) * (range_max - range_min) + range_min
    y = torch.tensor([[func(*x_i.tolist())] for x_i in x], dtype=torch.float32)

    if noise_std > 0:
        noise = torch.randn_like(y) * noise_std
        y += noise

    return x, y


def get_expression(model, input_vars):
    prior_expressions = copy.deepcopy(input_vars)
    model_str = ""

    for layer in model:
        new_expressions = []
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data
            bias = layer.bias.data
            for i in range(weights.shape[0]):
                terms = []
                for j in range(weights.shape[1]):
                    weight_val = weights[i, j].item()
                    if abs(weight_val) > 0.001 and prior_expressions[j] != "":
                        term = f"{weight_val:.4f} * ({prior_expressions[j]})"
                        terms.append(term)

                bias_val = bias[i].item()
                if abs(bias_val) > 0.001:
                    terms.append(f"{bias_val:.4f}")

                if terms:
                    new_expressions.append(" + ".join(terms))
                else:
                    new_expressions.append("")
        elif isinstance(layer, SymbolicActivation):
            i = 0

            for func in layer.funcs:
                if len(signature(func[1]).parameters) == 1:
                    if func[0] == "1":
                        new_expressions.append(func[0])
                    else:
                        if prior_expressions[i] != "":
                            new_expressions.append(func[0].replace("x", prior_expressions[i]))
                        else:
                            new_expressions.append("")
                    i += 1
                elif len(signature(func[1]).parameters) == 2:
                    expr1 = prior_expressions[i] if prior_expressions[i] != "" else "0"
                    expr2 = prior_expressions[i + 1] if prior_expressions[i + 1] != "" else "0"

                    if expr1 == "0" and expr2 == "0":
                        new_expressions.append("")
                    else:
                        new_expr = func[0].replace("x1", expr1).replace("x2", expr2)
                        new_expressions.append(new_expr)

                    i += 2
        prior_expressions = new_expressions

    model_str += prior_expressions[0]

    return model_str


def l05_regularization(model, eps=1e-8, bias=True, a=0.007):
    reg = 0.0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight
            abs_w = torch.abs(weights)

            large_mask = abs_w >= a
            small_mask = abs_w < a

            reg_large = torch.sqrt(abs_w[large_mask] + eps)

            w_small = weights[small_mask]
            smooth_expr = (-w_small ** 4) / (8 * a ** 3) + (3 * w_small ** 2) / (4 * a) + (3 * a / 8)
            reg_small = torch.sqrt(smooth_expr + eps)

            reg += reg_large.sum() + reg_small.sum()

            if bias and module.bias is not None:
                bias_vals = module.bias
                abs_b = torch.abs(bias_vals)

                large_mask_b = abs_b >= a
                small_mask_b = abs_b < a

                reg_large_b = torch.sqrt(abs_b[large_mask_b] + eps)

                b_small = bias_vals[small_mask_b]
                smooth_expr_b = (-b_small ** 4) / (8 * a ** 3) + (3 * b_small ** 2) / (4 * a) + (3 * a / 8)
                reg_small_b = torch.sqrt(smooth_expr_b + eps)

                reg += reg_large_b.sum() + reg_small_b.sum()

    return reg
