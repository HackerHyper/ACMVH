import math
import torch
import torch.nn as nn

from collections import OrderedDict




class MLP(nn.Module):
    def __init__(self, hidden_dim=[1000, 2048, 512], act=nn.Tanh()):
        super(MLP, self).__init__()
        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim

        orderedDict = OrderedDict()
        for i in range(len(hidden_dim) - 1):
            index = i + 1
            orderedDict['linear' + str(index)] = nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
            orderedDict['bn' + str(index)] = nn.BatchNorm1d(self.hidden_dim[i + 1])
            orderedDict['act' + str(index)] = act

        self.mlp = nn.Sequential(orderedDict)
        # self._initialize()

    def _initialize(self):
        nn.init.xavier_normal_(self.mlp.linear1.weight.data)
        nn.init.xavier_normal_(self.mlp.linear2.weight.data)

    def forward(self, x):
        return self.mlp(x)

