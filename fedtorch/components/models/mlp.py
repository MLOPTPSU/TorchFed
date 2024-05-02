# -*- coding: utf-8 -*-
from functools import reduce
import operator
import torch.nn as nn
import torch

from ..model_builder import MODEL

@MODEL.register_module()
class MLP(nn.Module):
    def __init__(self, dataset_config, hidden_size, drop_rate, robust=False):
        super(MLP, self).__init__()
        for p in ['num_classes', 'dimension']:
            if not hasattr(dataset_config, p):
                raise ValueError("'{}' should be specified in the dataset config for a MLP model!".format(p))
        # init
        if not isinstance(hidden_size, list):
            hidden_size = [hidden_size]
        self.num_layers = len(hidden_size)
        self.num_features = reduce(operator.mul,dataset_config.dimension)
        self.num_classes = dataset_config.num_classes
        self.image = isinstance(dataset_config.dimension, list)
        self.robust = robust

        if self.robust:
            self.noise = torch.nn.Parameter(torch.randn(self.num_features)*0.001, requires_grad=True)
        # define layers.
        for i in range(1, self.num_layers + 1):
            in_features = self.num_features if i == 1 else hidden_size[i-2]
            out_features = hidden_size[i-1]

            layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features, track_running_stats=False),
                nn.ReLU(),
                nn.Dropout(p=drop_rate))
            setattr(self, 'layer{}'.format(i), layer)

        self.fc = nn.Linear(hidden_size[-1], self.num_classes, bias=False)



    def forward(self, x):
        if self.image:
            x = x.view(-1, self.num_features)
        if self.robust:
            x += self.noise
        for i in range(1, self.num_layers + 1):
            x = getattr(self, 'layer{}'.format(i))(x)
        x = self.fc(x)
        return x

