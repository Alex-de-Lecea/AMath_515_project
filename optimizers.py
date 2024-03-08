import os
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
import torch
from torch.optim.optimizer import Optimizer
import copy

class ProxGD(Optimizer):
    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        prox_g (function): the proximal gradient function for the non-smooth function g
    """

    def __init__(self, params, lr=0.01, prox_g = None):
        defaults = dict(lr=lr, prox_g = prox_g)
        super(ProxGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            prox_g = group['prox_g']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                step = p.data - lr * grad
                p.data = prox_g(step, lr)

        return loss
    
class GD(Optimizer):
    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        prox_g (function): the proximal gradient function for the non-smooth function g
    """

    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
        super(GD, self).__init__(params, defaults)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data.add_(grad, alpha = -lr)

        return loss