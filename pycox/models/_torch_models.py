'''
Some torch models.
'''

import torch
from torch import nn

class ReluNet(nn.Module):
    '''Relu net with dropout and batch norm

    Parameters:
        input_size: Input size.
        n_layers: Number of layers.
        n_nodes: Size of each layer.
        dropout: Dropout rate. If `False`, no dropout.
        batch_norm: If use of batch norm.
    '''
    def __init__(self, input_size, n_layers, n_nodes, dropout=False, batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm

        net = [self.block(True)]
        net.extend([self.block() for _ in range(n_layers-1)])
        net.append(nn.Sequential(nn.Linear(self.n_nodes, 1, bias=False)))
        self.net = nn.Sequential(*net)

    def block(self, input=False):
        n_input = self.input_size if input else self.n_nodes
        mod = nn.Sequential(nn.Linear(n_input, self.n_nodes), nn.ReLU())
        if self.batch_norm:
            mod = nn.Sequential(mod, nn.BatchNorm1d(self.n_nodes))
        if self.dropout:
            mod = nn.Sequential(mod, nn.Dropout(self.dropout))
        return mod

    def forward(self, x):
        return self.net(x)


class FuncTorch(nn.Module):
    def __init__(self, func):
        self.func = func
        super().__init__()

    def forward(self, x):
        return self.func(x)


class _Expg(nn.Module):
    '''Class for adding exp to g model.'''
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x):
        return torch.exp(self.g(x))
