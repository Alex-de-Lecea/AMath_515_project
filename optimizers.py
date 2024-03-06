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
    """

    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(ProxGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ProxGD, self).__setstate__(state)

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
            #for p in group['params']:
                #p.data.add_(prox_g((lr * g), lr))

        return loss