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

def Armijo2(model, loss_fn, images, initial_lr, beta=0.5, c=1e-4):
    """
    Perform the Armijo line search to find the suitable step size.

    Parameters:
    - model: The PyTorch model being trained.
    - loss_fn: The loss function used for training.
    - images: The input data for the current batch.
    - initial_lr: The initial learning rate (step size).
    - beta: The factor to reduce the step size in each iteration.
    - c: The constant used in the Armijo condition.

    Returns:
    - alpha: The step size that satisfies the Armijo condition.
    """
    with torch.no_grad():
        # Save current model parameters
        orig_params = [param.clone() for param in model.parameters()]

        # Compute the loss and gradient at the current parameters
        model.zero_grad()
        reconst, _ = model(images)
        loss = loss_fn(reconst, images.flatten(start_dim=1))
        loss.backward()
        
        # Compute the gradient norm and the initial function value
        grad_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None).sqrt().item()
        f_x = loss.item()

        # Initialize step size
        alpha = initial_lr
        while True:
            # Update model parameters temporarily
            for param, orig_param in zip(model.parameters(), orig_params):
                param.data = orig_param - alpha * param.grad.data
            
            # Compute the loss at the new parameters
            with torch.no_grad():
                reconst, _ = model(images)
                f_x_alpha = loss_fn(reconst, images.flatten(start_dim=1)).item()
            
            # Check the Armijo condition
            if f_x_alpha <= f_x - c * alpha * grad_norm**2:
                break  # The condition is satisfied
            
            # Reduce the step size
            alpha *= beta
            # Restore original parameters before the next iteration
            for param, orig_param in zip(model.parameters(), orig_params):
                param.data = orig_param.data
        
        # Restore original parameters before returning
        for param, orig_param in zip(model.parameters(), orig_params):
            param.data = orig_param.data

    return alpha


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