
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

################################################################################
# LSTM
################################################################################

class SimpleLSTMCell(nn.Module):

    def __init__(self, input_size, output_size, orthogonal=False):
        super(SimpleLSTMCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.orthogonal = orthogonal
        self.lstm = nn.LSTMCell(input_size, output_size)

        if orthogonal:
            nn.init.orthogonal(self.lstm.weight_ih.data)
            nn.init.orthogonal(self.lstm.weight_hh.data)
        else:
            nn.init.xavier_normal(self.lstm.weight_ih.data)
            nn.init.xavier_normal(self.lstm.weight_hh.data)
        nn.init.constant(self.lstm.bias_ih.data, 0)
        nn.init.constant(self.lstm.bias_hh.data, 0)

    def reset(self, batch_size=1, cuda=False):
        if cuda:
            self.states = (Variable(torch.randn(batch_size, self.output_size)).float().cuda(), Variable(torch.randn(batch_size, self.output_size)).float().cuda())
        else:
            self.states = (Variable(torch.randn(batch_size, self.output_size)).float(), Variable(torch.randn(batch_size, self.output_size)).float())

    def detach(self):
        self.states = (self.states[0].detach(), self.states[1].detach())

    def forward(self, x):
        hx, cx = self.lstm(x, self.states)
        self.states = (hx, cx)
        return hx


class SimpleLSTM(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=128,
                 output_size=1,
                 layers=1,
                 orthogonal=False):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.orthogonal = orthogonal

        self.lstms = nn.ModuleList()
        self.lstms.append(SimpleLSTMCell(input_size, hidden_size, orthogonal=orthogonal))
        for i in range(self.layers-1):
            self.lstms.append(SimpleLSTMCell(hidden_size, hidden_size, orthogonal=orthogonal))
        self.fc = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal(self.fc.weight.data)
        nn.init.constant(self.fc.bias.data, 0)

        self.states = None

    def reset(self, batch_size=1, cuda=False):
        for i in range(len(self.lstms)):
            self.lstms[i].reset(batch_size, cuda)

    def detach(self):
        for i in range(len(self.lstms)):
            self.lstms[i].detach()

    def forward(self, x):
        """
        """
        for i in range(len(self.lstms)):
            x = self.lstms[i](x)
        out = self.fc(x)
        return out


################################################################################
# RNN
################################################################################


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class SimpleRNNCell(nn.Module):

    def __init__(self, input_size, output_size, orthogonal=False):
        super(SimpleRNNCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.orthogonal = orthogonal
        self.rnn = nn.RNNCell(input_size, output_size)

        if orthogonal:
            nn.init.orthogonal(self.rnn.weight_ih.data)
            nn.init.orthogonal(self.rnn.weight_hh.data)
        else:
            nn.init.xavier_normal(self.rnn.weight_ih.data)
            nn.init.xavier_normal(self.rnn.weight_hh.data)
        nn.init.constant(self.rnn.bias_ih.data, 0)
        nn.init.constant(self.rnn.bias_hh.data, 0)

        self.states = None

    def reset(self, batch_size=1, cuda=False):
        if cuda:
            self.states = Variable(torch.randn(batch_size, self.output_size)).float().cuda()
        else:
            self.states = Variable(torch.randn(batch_size, self.output_size)).float()

    def detach(self):
        self.states = self.states.detach()

    def forward(self, x):
        # print (x.size())
        hx = self.rnn(x, self.states)
        self.states = hx
        return hx


class SimpleRNN(nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=128,
                 output_size=1,
                 layers=1,
                 orthogonal=False):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.orthogonal = orthogonal

        self.rnns = nn.ModuleList()
        # the first layer will always be input-hidden
        self.rnns.append(SimpleRNNCell(input_size, hidden_size, orthogonal=orthogonal))
        for i in range(self.layers-1):
            # add req num of hidden rnns
            self.rnns.append(SimpleRNNCell(hidden_size, hidden_size, orthogonal=orthogonal))
        self.fc = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal(self.fc.weight.data)
        nn.init.constant(self.fc.bias.data, 0)

        self.states = None

    def reset(self, batch_size=1, cuda=False):
        for i in range(len(self.rnns)):
            self.rnns[i].reset(batch_size, cuda)

    def detach(self):
        for i in range(len(self.rnns)):
            self.rnns[i].detach()

    def forward(self, x):
        """
        """
        for i in range(len(self.rnns)):
            x = self.rnns[i](x)
        out = self.fc(x)
        return out
