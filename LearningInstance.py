import json
import os
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import symbolic_functions
from Seminar.symbolic_activation import SymbolicActivation
from Seminar.symbolic_functions import count_double
from Seminar.utility import get_expression, l05_regularization, generate_data, N_TRAIN, N_TEST, DOMAIN_TEST

var_names = ["x", "y", "z"]  # Variable names (for symbolic functions)


class LearningInstance:
    def __init__(self, results_dir, n_layers, reg_weight, learning_rate, n_epochs1, n_epochs2, freezing_epoch_distance):

        self.activation_funcs = [
            *[symbolic_functions.Identity()] * 4,
            *[symbolic_functions.Square()] * 4,
            *[symbolic_functions.Sin2Pi()] * 2,
            *[symbolic_functions.Exp()] * 2,
            *[symbolic_functions.SigmoidScaled()] * 2,
            *[symbolic_functions.Multiply()] * 2
        ]

        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2
        self.freezing_epoch_distance = freezing_epoch_distance
        self.print_interval = 1000
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir

        config_data = {
            "learning_rate": self.learning_rate,
            "print_interval": self.print_interval,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_names": [f.name for f in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }

        with open(os.path.join(self.results_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

    def run_experiment(self, target_function, func_name: str, attempts: int):
        func_dir = os.path.join(self.results_dir, func_name)
        os.makedirs(func_dir, exist_ok=True)
        print(func_name)
        expr_list, error_test_list = self.train(target_function, attempts, func_dir)

        error_expr_sorted = sorted(zip(error_test_list, expr_list))
        error_test_sorted = [err for err, _ in error_expr_sorted]
        expr_list_sorted = [expr for _, expr in error_expr_sorted]

        summary_path = os.path.join(self.results_dir, 'eq_summary.txt')
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{func_name}\n")
            for err, expr in zip(error_test_sorted, expr_list_sorted):
                f.write(f"[{err:.6f}]\t\t{str(expr)}\n")

    def train(self, func, atempts=1, func_dir='results/test'):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print("Use cuda:", use_cuda, "Device:", device)

        # Data generation
        x, y = generate_data(func, N_TRAIN, noise_std=1e-3)
        data, target = x.to(device), y.to(device)
        x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])
        test_data, test_target = x_test.to(device), y_test.to(device)

        x_dim = len(signature(func).parameters)
        width = len(self.activation_funcs)
        n_double = count_double(self.activation_funcs)
        hidden_layer_size = width + n_double

        expr_list = []
        error_test_list = []
        error_list = []

        error_test_list_atempt = []
        for atempt in range(atempts):
            # Define network
            model = nn.Linear(2, 2)
            loss_val = np.nan
            while np.isnan(loss_val):
                # Build network
                layers = []
                in_features = x_dim
                for _ in range(self.n_layers):
                    layers.append(nn.Linear(in_features, hidden_layer_size))
                    layers.append(SymbolicActivation(self.activation_funcs))
                    in_features = width
                layers.append(nn.Linear(in_features, 1))
                model = nn.Sequential(*layers).to(device)

                optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-10)
                criterion = nn.MSELoss()
                scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.1)

                # Learning all parameters
                for epoch in range(self.n_epochs1 + 2000):
                    optimizer.zero_grad()
                    outputs = model(data)
                    outputs = torch.clamp(outputs, min=-1e6, max=1e6)
                    mse_loss = criterion(outputs, target)
                    reg_loss = l05_regularization(model, eps=self.reg_weight)
                    loss = mse_loss + self.reg_weight * reg_loss
                    model(data)
                    loss.backward()
                    optimizer.step()

                    if epoch % self.print_interval == 0:
                        error_val = mse_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)

                        with torch.no_grad():
                            test_outputs = model(test_data)
                            test_loss = F.mse_loss(test_outputs, test_target)
                            error_test_val = test_loss.item()
                            error_test_list_atempt.append(error_test_val)
                        print(
                            f"atempt:{atempt} Epoch1: {epoch}\tTraining loss: {loss_val:.4f}\tTest error: {error_test_val:.4f}")
                        eks = get_expression(model, var_names[:x_dim])
                        if len(eks) < 151:
                            print(eks)
                        if np.isnan(loss_val) or loss_val > 10000:
                            break

                scheduler.step()
                freeze_threshold = 1e-1
                secondary_freeze_treshold = 1.5e-1
                # Freezing all zero parameters
                freeze_masks = {}
                for name, param in model.named_parameters():
                    with torch.no_grad():
                        mask = param.abs() >= freeze_threshold  # True for weights to keep training
                        param[~mask] = 0.0  # Zero out small weights
                        freeze_masks[name] = mask

                optimizer = optim.RMSprop(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=self.learning_rate * 0.1,
                    alpha=0.9,
                    eps=1e-10
                )

                # Learning with continued zrozen weigths
                for epoch in range(self.n_epochs2):
                    optimizer.zero_grad()
                    outputs = model(data)
                    outputs = torch.clamp(outputs, min=-1e6, max=1e6)
                    mse_loss = criterion(outputs, target)
                    reg_loss = l05_regularization(model, eps=self.reg_weight)
                    loss = mse_loss + self.reg_weight * reg_loss
                    loss.backward()
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= freeze_masks[name]
                    optimizer.step()
                    if (self.n_epochs2 - epoch <= 2000):
                        if (epoch % 100 == 50):
                            freeze_masks = {}
                            for name, param in model.named_parameters():
                                with torch.no_grad():
                                    mask = param.abs() >= secondary_freeze_treshold
                                    param[~mask] = 0.0
                                    freeze_masks[name] = mask
                            optimizer = optim.RMSprop(
                                filter(lambda p: p.requires_grad, model.parameters()),
                                lr=self.learning_rate * 0.1,
                                alpha=0.9,
                                eps=1e-10
                            )

                    if epoch % self.print_interval == 0:
                        error_val = mse_loss.item()
                        loss_val = loss.item()
                        error_list.append(error_val)

                        with torch.no_grad():
                            test_outputs = model(test_data)
                            test_loss = F.mse_loss(test_outputs, test_target)
                            error_test_val = test_loss.item()
                            error_test_list_atempt.append(error_test_val)

                        print(
                            f"atempt:{atempt} Epoch2: {epoch}\tTraining loss: {loss_val:.4f}\tTest error: {error_test_val:.4f}")
                        eks = get_expression(model, var_names[:x_dim])
                        if len(eks) < 151:
                            print(eks)
                        if np.isnan(loss_val) or loss_val > 10000:
                            break

            with torch.no_grad():
                expr = get_expression(model, var_names[:x_dim])
                print(expr)

            weights_file = os.path.join(func_dir, f'atempt{atempt}_weights.pt')
            torch.save(model.state_dict(), weights_file)

            meta = {
                "expr": expr,
                "error_list": error_list,
                "error_test": error_test_list_atempt
            }
            meta_file = os.path.join(func_dir, f'atempt{atempt}.json')
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

            expr_list.append(expr)
            error_test_list.append(error_test_list_atempt[-1])

        return expr_list, error_test_list
