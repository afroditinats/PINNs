import numpy as np
import torch
from torch import nn, autograd, optim, mean


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = self.create_network()
        self.bc = None
        self.ic = None
        self.cc = None
        self.ev = None
        self.pde = None
        self.visc = nn.Parameter(data=torch.tensor([-1.0]))
        self.network.register_parameter("visc", self.visc)

        self.lambda_bc = nn.Parameter(torch.tensor(1.0))
        self.lambda_ic = nn.Parameter(torch.tensor(1.0))
        self.lambda_pde = nn.Parameter(torch.tensor(1.0))

        self.adam_optimizer = optim.AdamW(self.network.parameters())
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.network.parameters(),
            lr=1,
            max_iter=2000,
            tolerance_grad=1e-128,
            tolerance_change=1e-128,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.val_loss = None

    def create_network(self):
        return nn.Sequential(
            nn.Linear(2, 20).double(),
            nn.Tanh().double(),
            nn.Linear(20, 20).double(),
            nn.Tanh().double(),
            nn.Linear(20, 20).double(),
            nn.Tanh().double(),
            nn.Linear(20, 20).double(),
            nn.Tanh().double(),
            nn.Linear(20, 1).double(),
        )

    def forward(self, inputs):
        return self.network(inputs[:, 0:2])

    def mse_loss(self, data):
        if len(data[0]) == 0:
            return torch.tensor([0.0], dtype=torch.double)
        return mean((self.forward(data[0]) - data[1]) ** 2)

    def phy_loss(self, pde):
        pde.requires_grad = True
        output = self.forward(pde)

        y_t = autograd.grad(output, pde, retain_graph=True, grad_outputs=torch.ones_like(output), create_graph=True)[0][:, 1]
        y_x = autograd.grad(output, pde, retain_graph=True, grad_outputs=torch.ones_like(output), create_graph=True)[0][:, 0]
        y_x2 = autograd.grad(y_x, pde, retain_graph=True, grad_outputs=torch.ones_like(y_x), create_graph=True)[0][:, 0]
        pde_loss = y_t + (torch.flatten(output) * y_x) - ((10**self.visc) * y_x2)
        return torch.square(pde_loss).mean()

    def update_weights(self, bc_loss, ic_loss, phy_loss):
        with torch.no_grad():
            loss_tensor = torch.tensor([bc_loss, ic_loss, phy_loss])
            softmax_weights = torch.nn.functional.softmax(-loss_tensor, dim=0) 

            self.lambda_bc.data = softmax_weights[0]
            self.lambda_ic.data = softmax_weights[1]
            self.lambda_pde.data = softmax_weights[2]

    def loss_fn(self, bc, ic, cc, val, pde):
        bc_loss = self.mse_loss(bc)
        ic_loss = self.mse_loss(ic)
        phy_loss = self.phy_loss(pde)
        val_loss = self.mse_loss(val)

        self.update_weights(bc_loss, ic_loss, phy_loss)

        total_loss = self.lambda_bc * bc_loss + self.lambda_ic * ic_loss + self.lambda_pde * phy_loss + self.mse_loss(cc)
        self.save_if_best(val_loss)

        return total_loss

    def save_if_best(self, val_loss):
        if self.val_loss is None or self.val_loss > val_loss:
            self.val_loss = val_loss
            torch.save(self.network.state_dict(), "best_adaptive.hdf5")

    def closure(self):
        self.lbfgs_optimizer.zero_grad()
        loss = self.loss_fn(self.bc, self.ic, self.cc, self.ev, self.pde)
        loss.backward()
        return loss

    def train_model(self, bc, ic, cc, val, pde, iterations):
        for _ in range(iterations):
            loss = self.loss_fn(bc, ic, cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()
            self.adam_optimizer.step()

        self.bc = bc
        self.ic = ic
        self.cc = cc
        self.ev = val
        self.pde = pde
        self.lbfgs_optimizer.step(self.closure)
