'''Hight level models that are easy to use.
'''

import torch
from torch import nn, optim
from .cox import CoxTime

class ReluNet(nn.Module):
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


class CoxTimeReluNet(CoxTime):
    def __init__(self, input_size, n_layers, n_nodes, dropout=False, batch_norm=True,
                 optimizer=None, device=None):
        g = ReluNet(input_size, n_layers, n_nodes, dropout=False, batch_norm=True)
        super().__init__(g, optimizer, device)
    
    def fit(self, df_train, duration_col, event_col=None, df_val=None, batch_size=64, epochs=10,
            n_workers=0, verbose=True, callbacks=None, compute_hazards=False, n_control=1,):
        '''
        TODO:
            Warn if input_size does not fit df! Give better warning than pytorch!!!!!!!!
        '''