# -*- coding: utf-8 -*-
from functools import reduce
import operator
import torch.nn as nn

from ..model_builder import MODEL

@MODEL.register_module()
class RNN(nn.Module):
    def __init__(self, dataset_config, hidden_size, output_size, batch_size, n_layers=1):
        super(RNN, self).__init__()
        for p in ['dimension']:
            if not hasattr(dataset_config, p):
                raise ValueError("'{}' should be specified in the dataset config for a RNN model!".format(p))
        self.input_size = reduce(operator.mul,dataset_config.dimension)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        
        self.encoder = nn.Embedding(self.input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        

        self.init_hidden(self.batch_size)
    
    def forward(self, input):
        if self.hidden.size(1) != input.size(0):
            self.init_hidden(input.size(0))
        if self.hidden.device != input.device:
            self.hidden = self.hidden.to(input.device)
        input = self.encoder(input)
        output, h = self.gru(input, self.hidden.detach())
        self.hidden.data = h.data
        output = self.decoder(output)
        return output.permute(0,2,1)
    
    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size
        weight = next(self.parameters())
        self.hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_size)
        return