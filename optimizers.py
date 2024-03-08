import os
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
import torch
from torch.optim.optimizer import Optimizer
import copy

def Armijo(x_k, func_f, grad_f, t = 0.8, c = 0.01, beta = 0.5):
    while (func_f(x_k + t*grad_f) - func_f(x_k) > c*t*np.matmul(np.tranpose(grad_f), grad_f)):
        t = beta* t
    return t / beta

def Armijo2(grad_f, error, reconst, images, t = 0.8, c = 0.01, beta = 0.5):
    current_loss = error(reconst, images.flatten(start_dim = 1))
    new_loss = error(reconst + t*grad_f, images.flatten(start_dim = 1))
    while (new_loss - current_loss > c*t*(norm(grad_f)**2)):
        t = beta * t
        new_loss = error(reconst + t*grad_f, images.flatten(start_dim = 1))

    return t/beta


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