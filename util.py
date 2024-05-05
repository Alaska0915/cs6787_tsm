import numpy as np
import copy
import matplotlib.pyplot as plt

import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader


def get_parameter_num(model, trainable = True):
    if trainable:
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num =  sum(p.numel() for p in model.parameters())
    return num


def load_CIFAR10_dataset():
    transform = torchvision.transforms.ToTensor()
    cifar10_path = './data'
    cifar10_train = torchvision.datasets.CIFAR10(root=cifar10_path, train=True, transform=transform, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=cifar10_path, train=False, transform=transform)
    cifar10_splitted_train, cifar10_validation = torch.utils.data.random_split(
        cifar10_train, [45000, 5000], generator=torch.Generator().manual_seed(42))
    return (cifar10_train, cifar10_test, cifar10_splitted_train, cifar10_validation)


def construct_dataloaders(dataset, batch_size, shuffle_train=True):
    train_dataset, test_dataset, splitted_train_dataset, validation_dataset = dataset
    train_dataloader = DataLoader(train_dataset,
                                batch_size = batch_size,
                                shuffle = shuffle_train,)
    test_dataloader = DataLoader(test_dataset,
                                batch_size = 100,
                                shuffle = False,)
    splitted_train_dataloader = DataLoader(splitted_train_dataset,
                                batch_size = batch_size,
                                shuffle = shuffle_train,)
    validation_dataloader = DataLoader(validation_dataset,
                                batch_size = 100,
                                shuffle = False,)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['test'] = test_dataloader
    dataloaders['splitted_train'] = splitted_train_dataloader
    dataloaders['validation'] = validation_dataloader
    return dataloaders


def evaluate_model(dataloader, model, loss_fn):
    loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            loss += loss_fn(y_hat, y).item()
            accuracy += (y_hat.argmax(1) == y).type(torch.float).sum().item()
    loss = loss / len(dataloader.dataset)
    accuracy = accuracy / len(dataloader.dataset)
    return (loss, accuracy)

