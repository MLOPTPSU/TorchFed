# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F

from ..model_builder import MODEL

@MODEL.register_module()
class Lenet(nn.Module):
    def __init__(self,dataset_config):
        super(Lenet, self).__init__()
        for p in ['num_classes', 'dimension']:
            if not hasattr(dataset_config, p):
                raise ValueError("'{}' should be specified in the dataset config for a Lenet model!".format(p))
        self.num_classes = dataset_config.num_classes
        self.dimension = dataset_config.dimension
        self.num_channels = 1 if len(dataset_config.dimension) == 2 else dataset_config.dimension[0]
        self.conv1 = nn.Conv2d(self.num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.rep_out_dim = self._decide_output_representation_size()
        self.fc1 = nn.Linear(self.rep_out_dim, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    
    def _decide_output_representation_size(self):
        dim = self.dimension[1] // 4 - 3
        return 50 * dim * dim
        


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.rep_out_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

