# Copyright (c) FedTorch. All rights reserved.
from functools import reduce
import operator
import torch
import torch.nn as nn

from ..model_builder import MODEL

@MODEL.register_module()
class Least_square(nn.Module):

    def __init__(self, dataset_config, bias=True, robust=False):
        super(Least_square, self).__init__()
        for p in ['dimension', 'num_classes']:
            if not hasattr(dataset_config, p):
                raise ValueError("'{}' should be specified in the dataset config for a regression model!".format(p))
        self.num_features = reduce(operator.mul,dataset_config.dimension)
        if dataset_config.num_classes > 1:
            raise ValueError("The regression model is not suitable for multiclass datasets!")
        self.bias = bias
        self.robust = robust

        # define noise params in robust mode
        if self.robust:
            self.noise = torch.nn.Parameter(torch.randn(self.num_features)*0.001, requires_grad=True)
        # define layers.
        self.fc = nn.Linear(
            in_features=self.num_features,
            out_features=1, bias=True)

    def forward(self, x):
        if self.robust:
            x += self.noise
        x = self.fc(x)
        return x
