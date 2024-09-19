"""This file contains functions for constructing various loss functions that could be used during training."""


import torch


def get_crossentropy_loss(class_weights):
    if isinstance(class_weights, list):
        class_weights = torch.tensor(class_weights)
    return torch.nn.CrossEntropyLoss(weight=class_weights)


def get_binary_crossentropy_loss():
    return torch.nn.BCELoss()


def get_custom_binary_crossentropy_loss(class_weights):

    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1-1e-7)
        bce = - class_weights[1] * target * torch.log(input) - (1 - target) * class_weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss
