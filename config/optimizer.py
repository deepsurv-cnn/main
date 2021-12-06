#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.optim as optim



def Optimizer(optimizer_name, model, lr):
    """
    Usage:
    from lib.criterion import Optimizer
    optimizer = Optimizer(model, 'SGD', lr), where like model = Model('ResNet', 2), lr = args['lr']
    """

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)

    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)

    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    else:
        print('No specified optimizer: {}.'.format(optimizer_name))
        exit()

    return optimizer


# ----- EOF -----
