import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class VanillaGRUCell(nn.Module):

    def __init__(self, input_size, output_size):
        super(VanillaGRUCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gru = nn.GRUCell(input_size, output_size)

        nn.init.xavier_normal(self.gru.weight_ih.data)
        nn.init.orthogonal(self.gru.weight_hh.data)
        nn.init.constant(self.gru.bias_ih.data, 0)
        nn.init.constant(self.gru.bias_hh.data, 0)

    def reset(self, batch_size=1, cuda=True):
        if cuda:
            self.recurrent_state = (Variable(torch.zeros(batch_size, self.output_size)).float().cuda())
        else:
            self.recurrent_state = (Variable(torch.zeros(batch_size, self.output_size)).float())

    def forward(self, x):
        hx = self.gru(x, self.recurrent_state)
        self.recurrent_state = hx
        return hx

class VanillaGRU(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=128,
                 output_size=1,
                 layers=1):
        super(VanillaGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers

        self.grus = nn.ModuleList()
        self.grus.append(VanillaGRUCell(input_size, hidden_size))
        for i in range(self.layers-1):
            self.grus.append(VanillaGRUCell(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal(self.fc.weight.data)
        nn.init.constant(self.fc.bias.data, 0)

        self.recurrent_state = None

    def reset(self, batch_size=1, cuda=True):
        for i in range(len(self.grus)):
            self.grus[i].reset(batch_size, cuda)

    def forward(self, x):

        for i in range(len(self.grus)):
            x = self.grus[i](x)
        out = self.fc(x)
        return out
