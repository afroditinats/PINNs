import numpy as np
import torch
from torch import nn, autograd, optim
from sklearn.metrics import mean_squared_error
from collections import deque

#mse-= same for all models 

def mse_loss(self, data):
    if len(data[0]) == 0:
        return torch.tensor([0.], dtype=torch.float64)
    output = self.forward(data[0])
    return torch.mean((output - data[1]) ** 2)

class EarlyStoppingCriterion(nn.Module):
    #uss fixed thresholds for rmse and p. error and stops when both are below target values.
    def __init__(self, target_rmse=1e-9, target_param_error=1e-9):
        super().__init__()
        self.network = self.create_network()
        self.visc = nn.Parameter(data=torch.tensor([-1.0]))
        self.network.register_parameter("visc", self.visc)
        self.adam_optimizer = optim.AdamW(self.network.parameters())
        self.lbfgs_optimizer = optim.LBFGS(self.network.parameters(), lr=1, max_iter=2000,
                                           tolerance_grad=1e-128, tolerance_change=1e-128,
                                           history_size=50, line_search_fn="strong_wolfe")
        self.target_rmse = target_rmse
        self.target_param_error = target_param_error

    def create_network(self):
        layers = [nn.Linear(2, 20).double(), nn.Tanh().double()]
        for _ in range(3):
            layers += [nn.Linear(20, 20).double(), nn.Tanh().double()]
        layers.append(nn.Linear(20, 1).double())
        return nn.Sequential(*layers)

    def forward(self, x): return self.network(x[:, 0:2])
    def mse_loss(self, data): return mse_loss(self, data)

    def phy_loss(self, x):
        x.requires_grad = True
        y = self.forward(x)
        y_t = autograd.grad(y, x, torch.ones_like(y), retain_graph=True, create_graph=True)[0][:, 1]
        y_x = autograd.grad(y, x, torch.ones_like(y), retain_graph=True, create_graph=True)[0][:, 0]
        y_xx = autograd.grad(y_x, x, torch.ones_like(y_x), retain_graph=True, create_graph=True)[0][:, 0]
        return torch.mean((y_t + y.flatten() * y_x - (10**self.visc) * y_xx) ** 2)

    def loss_fn(self, bc, ic, cc, val, pde):
        return self.mse_loss(bc) + self.mse_loss(ic) + self.mse_loss(cc) + self.phy_loss(pde), self.mse_loss(val)

    def closure(self):         #helper function necessary for L-BFGS.
        self.lbfgs_optimizer.zero_grad()
        loss, _ = self.loss_fn(self.bc, self.ic, self.cc, self.ev, self.pde)
        loss.backward()
        return loss

    def train_model(self, bc, ic, cc, val, pde, iterations):
        for i in range(iterations):
            loss, val_loss = self.loss_fn(bc, ic, cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()
            self.adam_optimizer.step()
            pred = self.forward(val[0])
            rmse = mean_squared_error(pred.detach().numpy(), val[1].numpy())
            param = 10 ** self.visc.item()
            param_error = mean_squared_error([param], [0.01 / np.pi])
            if rmse < self.target_rmse and param_error < self.target_param_error:
                print(f"Early stopping: target RMSE and param error reached at iteration {i}")
                break
        self.bc, self.ic, self.cc, self.ev, self.pde = bc, ic, cc, val, pde
        self.lbfgs_optimizer.step(self.closure)


class PatienceStoppingModel(EarlyStoppingCriterion):
    #checks if we have significant progress in each step. --if RMSE and parameter error stay close for 100 steps ->break.
    def __init__(self, patience=100, min_delta=1e-6): #min_delta:= the threshold for the validation loss (to check if there is improvbent)
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta

    def train_model(self, bc, ic, cc, val, pde, iterations):
        best_loss = float("inf")
        wait = 0
        for i in range(iterations):
            loss, val_loss = self.loss_fn(bc, ic, cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()
            self.adam_optimizer.step()
            current_loss = val_loss.item()
            if best_loss - current_loss > self.min_delta:
                best_loss = current_loss
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                print(f"Patience stopping at iteration {i}")
                break
        self.bc, self.ic, self.cc, self.ev, self.pde = bc, ic, cc, val, pde
        self.lbfgs_optimizer.step(self.closure)


class GradientNormStoppingModel(EarlyStoppingCriterion):
    #monitors gradient during training and stops early if gradients become too small.
    #the threshold is applied only in the Adam optimizer. (L-BFGS is not affected by this criterion) 
    def __init__(self, grad_threshold=1e-6): #grad_threshold:= the threshold for the gradient norm /to check if there is improvement
        super().__init__()
        self.grad_threshold = grad_threshold

    def train_model(self, bc, ic, cc, val, pde, iterations):
        for i in range(iterations):
            loss, val_loss = self.loss_fn(bc, ic, cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()
            total_norm = 0
            for p in self.network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.adam_optimizer.step()
            if total_norm < self.grad_threshold:
                print(f"Gradient norm stopping at iteration {i}, norm = {total_norm:.2e}")
                break

        self.bc, self.ic, self.cc, self.ev, self.pde = bc, ic, cc, val, pde
        self.lbfgs_optimizer.step(self.closure)

class HybridStoppingModel(EarlyStoppingCriterion):
    def __init__(self, patience=100, min_delta=1e-6, slope_threshold=1e-7, grad_threshold=1e-6, window_size=10):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.slope_threshold = slope_threshold
        self.grad_threshold = grad_threshold
        self.loss_window = deque(maxlen=window_size)

    def train_model(self, bc, ic, cc, val, pde, iterations):
        #combines multiple stopping criteria -all the above:
        #--Conditions for stopping:--
        #valid. loss improvement /patience & min_delta threshold
        #slope over recent steps /slope_threshold 
        #gradient norm /grad_threshold 
        best_loss = float("inf")
        wait = 0

        for i in range(iterations):
            loss, val_loss = self.loss_fn(bc, ic, cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()

            total_norm = 0
            for p in self.network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.adam_optimizer.step()

            current_loss = val_loss.item()
            if best_loss - current_loss > self.min_delta:
                best_loss = current_loss
                wait = 0
            else:
                wait += 1

            self.loss_window.append(current_loss)
            slope = 0
            if len(self.loss_window) == self.loss_window.maxlen:
                x_vals = np.arange(len(self.loss_window))
                y_vals = np.array(self.loss_window)
                slope = np.polyfit(x_vals, y_vals, 1)[0]
            if wait >= self.patience and abs(slope) < self.slope_threshold and total_norm < self.grad_threshold:
                print(f"Hybrid stop at iteration {i}: no improvement + flat slope + tiny gradient")
                break
        self.bc, self.ic, self.cc, self.ev, self.pde = bc, ic, cc, val, pde
        self.lbfgs_optimizer.step(self.closure)

print("ok modules_esc")



"""

slope:***small slope <=> validation loss has plateaued (almost flat)***
slope = np.polyfit(x_vals, y_vals, 1)[0]
linear regression over the most recent validation losses(store them in self.loss_window) to find the slope of the loss curve.

grad_threshold: computes the L2 norm of the gradients/ (gradiest includes all trainable parameters weights etc that belong to the NN, including the viscosit
param_norm = p.grad.data.norm(2)
total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
!the partial derivative of the loss function with respect to each trainable parameter of the neural network (including viscosity).

window_size: number of most recent validation loss values used to compute the slope
self.loss_window = deque(maxlen=window_size)
Only computes slope when we have 10 data points => gives more stable estimation of the loss


"""