# Copyright (c) FedTorch. All rights reserved.
from functools import reduce
import operator
import torch
import torch.nn as nn

from ..model_builder import MODEL

@MODEL.register_module()
class LogisticRegression(torch.nn.Module):

    def __init__(self, dataset_config, bias=True, robust=False):
        super(LogisticRegression, self).__init__()
        for p in ['num_classes', 'dimension']:
            if not hasattr(dataset_config, p):
                raise ValueError("'{}' should be specified in the dataset config for a logistic regression model!".format(p))
        self.num_features = reduce(operator.mul,dataset_config.dimension)
        self.num_classes = dataset_config.num_classes
        self.bias = bias
        self.robust = robust
        self.image = isinstance(dataset_config.dimension, list)

        # define noise params in robust mode
        if self.robust:
            self.noise = torch.nn.Parameter(torch.randn(self.num_features)*0.001, requires_grad=True)
        # define layers.
        self.fc = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes, bias=self.bias)

        self._weight_initialization()

    def forward(self, x):
        # We don't need the softmax layer here since CrossEntropyLoss already
        # uses it internally.
        if self.image:
            x = x.view(-1, self.num_features)
        if self.robust:
            x += self.noise
        x = self.fc(x)
        return x

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(mean=0, std=0.01)
                m.weight.data.zero_()
                m.bias.data.zero_()

